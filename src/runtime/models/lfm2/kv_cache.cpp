/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lfm2/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

bool all_ok(const std::vector<DeviceTensor>& tensors) {
    return std::all_of(tensors.begin(), tensors.end(),
                       [](const DeviceTensor& tensor) { return tensor.ok(); });
}

void validate_scalar(TrtModule& module, const std::string& name) {
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kInt32 ||
        module.tensor_shape(name) != std::vector<int64_t>{1}) {
        throw std::runtime_error("LFM2 native KV input '" + name + "' must be int32 [1]");
    }
}

} // namespace

Lfm2KvCache::Lfm2KvCache(int32_t num_layers, int32_t max_length, int32_t num_kv_heads,
                         int32_t head_dim, cudaStream_t stream, DType dtype, Lfm2KvCacheNames names)
    : num_layers_(num_layers), max_length_(max_length), num_kv_heads_(num_kv_heads),
      head_dim_(head_dim), stream_(stream), dtype_(dtype), names_(std::move(names)) {
    if (num_layers_ <= 0 || max_length_ <= 0 || num_kv_heads_ <= 0 || head_dim_ <= 0 ||
        num_kv_heads_ > std::numeric_limits<int32_t>::max() / head_dim_) {
        throw std::invalid_argument("Lfm2KvCache requires positive, representable geometry");
    }
    kv_dim_ = num_kv_heads_ * head_dim_;

    if (names_.cache_k.empty()) {
        names_.cache_k.reserve(static_cast<std::size_t>(num_layers_));
        names_.cache_v.reserve(static_cast<std::size_t>(num_layers_));
        names_.present_k.reserve(static_cast<std::size_t>(num_layers_));
        names_.present_v.reserve(static_cast<std::size_t>(num_layers_));
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            const auto suffix = "_" + std::to_string(layer);
            names_.cache_k.push_back("cache_k" + suffix);
            names_.cache_v.push_back("cache_v" + suffix);
            names_.present_k.push_back("present_k" + suffix);
            names_.present_v.push_back("present_v" + suffix);
        }
    }
    const auto expected = static_cast<std::size_t>(num_layers_);
    if (names_.cache_k.size() != expected || names_.cache_v.size() != expected ||
        names_.present_k.size() != expected || names_.present_v.size() != expected) {
        throw std::invalid_argument(
            "Lfm2KvCache tensor-name count does not match attention layers");
    }

    cache_k_.reserve(expected);
    cache_v_.reserve(expected);
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        cache_k_.emplace_back(std::vector<int64_t>{max_length_, kv_dim_}, dtype_, stream_);
        cache_v_.emplace_back(std::vector<int64_t>{max_length_, kv_dim_}, dtype_, stream_);
    }
    reset();
}

void Lfm2KvCache::validate_contract(TrtModule& module) const {
    validate_scalar(module, names_.cache_write_indices);
    validate_scalar(module, names_.key_value_lengths);
    validate_scalar(module, names_.position_id);
    const std::vector<int64_t> expected_shape{1, num_kv_heads_, max_length_, head_dim_};
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        const auto validate_pair = [&](const std::string& cache, const std::string& present) {
            if (!module.has_input(cache) || !module.has_output(present)) {
                throw std::runtime_error("LFM2 native KV engine is missing pair '" + cache + "'/'" +
                                         present + "'");
            }
            if (module.tensor_shape(cache) != expected_shape ||
                module.tensor_shape(present) != expected_shape) {
                throw std::runtime_error(
                    "LFM2 native KV tensors must use [1,Hkv,capacity,head_dim]");
            }
            if (module.tensor_dtype(cache) != dtype_ || module.tensor_dtype(present) != dtype_) {
                throw std::runtime_error("LFM2 native KV tensor dtype does not match precision");
            }
        };
        validate_pair(names_.cache_k[index], names_.present_k[index]);
        validate_pair(names_.cache_v[index], names_.present_v[index]);
    }
}

void Lfm2KvCache::bind_to(TrtModule& module) {
    validate_contract(module);
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        module.bind_external(names_.cache_k[index], cache_k_[index].data());
        module.bind_external(names_.cache_v[index], cache_v_[index].data());
        if (module.device_ptr(names_.cache_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.present_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.cache_v[index]) != cache_v_[index].data() ||
            module.device_ptr(names_.present_v[index]) != cache_v_[index].data()) {
            throw std::runtime_error("LFM2 native KV cache/present aliasing was not preserved");
        }
    }
}

void Lfm2KvCache::prepare_step(TensorMap& inputs, int32_t seq_len) {
    if (seq_len <= 0)
        throw std::invalid_argument("LFM2 KV step length must be positive");
    if (seq_len > max_length_ - position_)
        throw std::runtime_error("LFM2 sequence exceeds fixed native KV cache capacity");

    cache_write_index_ = position_;
    key_value_length_ = position_ + seq_len;
    inputs[names_.cache_write_indices] = Tensor{&cache_write_index_, {1}, DType::kInt32};
    inputs[names_.key_value_lengths] = Tensor{&key_value_length_, {1}, DType::kInt32};

    // LFM2 always uses rotary position embeddings for the dense checkpoints.
    // The v1 engine is single-token, but keeping the vector-shaped contract
    // makes the state correct if a family-owned multi-token engine is added.
    positions_.resize(static_cast<std::size_t>(seq_len));
    for (int32_t index = 0; index < seq_len; ++index)
        positions_[static_cast<std::size_t>(index)] = position_ + index;
    inputs[names_.position_id] =
        Tensor{positions_.data(), {static_cast<int64_t>(seq_len)}, DType::kInt32};
}

void Lfm2KvCache::advance(int32_t n_tokens) {
    if (n_tokens <= 0 || n_tokens > max_length_ - position_)
        throw std::runtime_error("Invalid LFM2 native KV advancement");
    position_ += n_tokens;
}

void Lfm2KvCache::reset() {
    // IKVCacheUpdate and key_value_lengths make stale capacity invisible.
    position_ = 0;
    cache_write_index_ = 0;
    key_value_length_ = 0;
}

std::size_t Lfm2KvCache::device_memory_bytes() const {
    std::size_t total = 0;
    for (const auto& tensor : cache_k_)
        total += tensor.nbytes();
    for (const auto& tensor : cache_v_)
        total += tensor.nbytes();
    return total;
}

bool Lfm2KvCache::ok() const {
    const auto expected = static_cast<std::size_t>(num_layers_);
    return cache_k_.size() == expected && cache_v_.size() == expected && all_ok(cache_k_) &&
           all_ok(cache_v_);
}

} // namespace trtmc
