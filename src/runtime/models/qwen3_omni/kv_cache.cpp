/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen3_omni/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

void validate_scalar(TrtModule& module, const std::string& name) {
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kInt32 ||
        module.tensor_shape(name) != std::vector<int64_t>{1}) {
        throw std::runtime_error("Qwen3-Omni native KV input '" + name + "' must be int32 [1]");
    }
}

bool valid_cache_shape(const std::vector<int64_t>& shape, int32_t capacity, int32_t kv_dim) {
    return shape.size() == 4 && shape[0] == 1 && shape[1] > 0 &&
           shape[2] == static_cast<int64_t>(capacity) && shape[3] > 0 &&
           shape[1] * shape[3] == kv_dim;
}

bool tensors_ok(const std::vector<DeviceTensor>& tensors) {
    for (const auto& tensor : tensors) {
        if (!tensor.ok())
            return false;
    }
    return true;
}

} // namespace

Qwen3OmniKvCache::Qwen3OmniKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim,
                                   cudaStream_t stream, DType cache_dtype,
                                   Qwen3OmniKvCacheNames names)
    : num_layers_(num_layers), max_length_(max_length), kv_dim_(kv_dim), cache_dtype_(cache_dtype),
      names_(std::move(names)) {
    if (num_layers <= 0 || max_length <= 0 || kv_dim <= 0)
        throw std::invalid_argument("Qwen3-Omni native KV dimensions must be positive");

    if (names_.cache_k.empty()) {
        for (int32_t layer = 0; layer < num_layers; ++layer) {
            const auto suffix = "_" + std::to_string(layer);
            names_.cache_k.push_back("cache_k" + suffix);
            names_.cache_v.push_back("cache_v" + suffix);
            names_.present_k.push_back("present_k" + suffix);
            names_.present_v.push_back("present_v" + suffix);
        }
    }
    const auto expected = static_cast<std::size_t>(num_layers);
    if (names_.cache_k.size() != expected || names_.cache_v.size() != expected ||
        names_.present_k.size() != expected || names_.present_v.size() != expected) {
        throw std::invalid_argument("Qwen3-Omni native KV tensor name count mismatch");
    }

    cache_k_.reserve(expected);
    cache_v_.reserve(expected);
    for (int32_t layer = 0; layer < num_layers; ++layer) {
        cache_k_.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype, stream);
        if (!cache_k_.back().ok())
            return;
        cache_v_.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype, stream);
        if (!cache_v_.back().ok())
            return;
    }
    reset();
}

void Qwen3OmniKvCache::validate_native_contract(TrtModule& module) const {
    validate_scalar(module, names_.cache_write_indices);
    validate_scalar(module, names_.key_value_lengths);
    if (module.has_input("attention_mask"))
        throw std::runtime_error("Qwen3-Omni native KV engine must not expose attention_mask");
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        for (const auto& pair : {std::pair{names_.cache_k[index], names_.present_k[index]},
                                 std::pair{names_.cache_v[index], names_.present_v[index]}}) {
            if (!module.has_input(pair.first) || !module.has_output(pair.second))
                throw std::runtime_error("Qwen3-Omni native KV cache/present pair is missing");
            const auto shape = module.tensor_shape(pair.first);
            if (!valid_cache_shape(shape, max_length_, kv_dim_) ||
                module.tensor_shape(pair.second) != shape ||
                module.tensor_dtype(pair.first) != cache_dtype_ ||
                module.tensor_dtype(pair.second) != cache_dtype_) {
                throw std::runtime_error("Qwen3-Omni native KV cache/present must share BF16 "
                                         "[1,num_kv_heads,capacity,head_dim]");
            }
        }
    }
}

void Qwen3OmniKvCache::bind_native_cache(TrtModule& module) {
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        module.bind_external(names_.cache_k[index], cache_k_[index].data());
        module.bind_external(names_.cache_v[index], cache_v_[index].data());
        if (module.device_ptr(names_.cache_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.present_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.cache_v[index]) != cache_v_[index].data() ||
            module.device_ptr(names_.present_v[index]) != cache_v_[index].data()) {
            throw std::runtime_error(
                "Qwen3-Omni native KV engine did not preserve cache/present aliasing");
        }
    }
}

void Qwen3OmniKvCache::bind_to(TrtModule& module) {
    has_position_input_ = module.has_input(names_.position_id);
    validate_native_contract(module);
    bind_native_cache(module);
}

void Qwen3OmniKvCache::bind_cache_inputs(TrtModule& module) {
    bind_to(module);
}

void Qwen3OmniKvCache::write_position_input(TensorMap& inputs, int32_t seq_len) {
    if (!has_position_input_)
        return;
    position_buf_.resize(static_cast<std::size_t>(seq_len));
    for (int32_t offset = 0; offset < seq_len; ++offset)
        position_buf_[static_cast<std::size_t>(offset)] = position_ + offset;
    inputs[names_.position_id] =
        Tensor{position_buf_.data(), {static_cast<int64_t>(seq_len)}, DType::kInt32};
}

void Qwen3OmniKvCache::prepare_step(TensorMap& inputs, int32_t seq_len) {
    if (seq_len <= 0)
        throw std::invalid_argument("Qwen3-Omni native KV step length must be positive");
    if (seq_len > max_length_ - position_)
        throw std::runtime_error("Qwen3-Omni sequence exceeds official context capacity");
    write_position_input(inputs, seq_len);
    cache_write_index_ = position_;
    key_value_length_ = position_ + seq_len;
    inputs[names_.cache_write_indices] = Tensor{&cache_write_index_, {1}, DType::kInt32};
    inputs[names_.key_value_lengths] = Tensor{&key_value_length_, {1}, DType::kInt32};
}

void Qwen3OmniKvCache::validate_native_aliases(const std::vector<const void*>& present_k,
                                               const std::vector<const void*>& present_v) const {
    if (present_k.size() != static_cast<std::size_t>(num_layers_) ||
        present_v.size() != static_cast<std::size_t>(num_layers_)) {
        throw std::runtime_error("Qwen3-Omni native KV pointer count mismatch");
    }
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        if (present_k[index] != cache_k_[index].data() ||
            present_v[index] != cache_v_[index].data()) {
            throw std::runtime_error("Qwen3-Omni native prefill outputs must alias the KV cache");
        }
    }
}

void Qwen3OmniKvCache::append_prefill_kv(const std::vector<const void*>& present_k,
                                         const std::vector<const void*>& present_v,
                                         int32_t seq_len) {
    if (seq_len <= 0 || seq_len > max_length_ - position_)
        throw std::runtime_error("Qwen3-Omni native prefill exceeds context capacity");
    validate_native_aliases(present_k, present_v);
    position_ += seq_len;
}

void Qwen3OmniKvCache::advance(int32_t n_tokens) {
    if (n_tokens <= 0 || n_tokens > max_length_ - position_)
        throw std::runtime_error("Qwen3-Omni native decode exceeds context capacity");
    position_ += n_tokens;
}

void Qwen3OmniKvCache::reset() {
    position_ = 0;
    cache_write_index_ = 0;
    key_value_length_ = 0;
}

std::size_t Qwen3OmniKvCache::device_memory_bytes() const {
    std::size_t total = 0;
    for (const auto& tensor : cache_k_)
        total += tensor.nbytes();
    for (const auto& tensor : cache_v_)
        total += tensor.nbytes();
    return total;
}

bool Qwen3OmniKvCache::ok() const {
    return cache_k_.size() == static_cast<std::size_t>(num_layers_) &&
           cache_v_.size() == static_cast<std::size_t>(num_layers_) && tensors_ok(cache_k_) &&
           tensors_ok(cache_v_);
}

} // namespace trtmc
