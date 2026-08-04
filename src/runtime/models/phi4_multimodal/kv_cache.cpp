/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/phi4_multimodal/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

void validate_scalar_input(TrtModule& module, const std::string& name) {
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kInt32 ||
        module.tensor_shape(name) != std::vector<int64_t>{1}) {
        throw std::runtime_error("Phi-4 Multimodal native KV input '" + name +
                                 "' must be int32 [1]");
    }
}

bool valid_cache_shape(const std::vector<int64_t>& shape, int32_t max_length, int32_t kv_dim) {
    return shape.size() == 4 && shape[0] == 1 && shape[2] == static_cast<int64_t>(max_length) &&
           shape[1] > 0 && shape[3] > 0 && shape[1] * shape[3] == kv_dim;
}

void validate_cache_pair(TrtModule& module, const std::string& cache_name,
                         const std::string& present_name, int32_t max_length, int32_t kv_dim,
                         DType cache_dtype) {
    if (!module.has_input(cache_name) || !module.has_output(present_name)) {
        throw std::runtime_error(
            "Phi-4 Multimodal native KV engine is missing cache/present pair '" + cache_name +
            "'/'" + present_name + "'");
    }
    const auto cache_shape = module.tensor_shape(cache_name);
    if (!valid_cache_shape(cache_shape, max_length, kv_dim) ||
        module.tensor_shape(present_name) != cache_shape) {
        throw std::runtime_error(
            "Phi-4 Multimodal native KV cache/present tensors must share static "
            "[1,Hkv,max_length,D] shape");
    }
    if (module.tensor_dtype(cache_name) != cache_dtype ||
        module.tensor_dtype(present_name) != cache_dtype) {
        throw std::runtime_error(
            "Phi-4 Multimodal native KV cache dtype does not match model precision");
    }
}

void populate_default_cache_names(Phi4MultimodalKvCacheNames& names, int32_t num_layers) {
    if (!names.cache_k.empty())
        return;

    const auto count = static_cast<std::size_t>(num_layers);
    names.cache_k.reserve(count);
    names.cache_v.reserve(count);
    names.present_k.reserve(count);
    names.present_v.reserve(count);
    for (int32_t layer = 0; layer < num_layers; ++layer) {
        const std::string suffix = "_" + std::to_string(layer);
        names.cache_k.push_back("cache_k" + suffix);
        names.cache_v.push_back("cache_v" + suffix);
        names.present_k.push_back("present_k" + suffix);
        names.present_v.push_back("present_v" + suffix);
    }
}

void validate_cache_name_count(const Phi4MultimodalKvCacheNames& names, int32_t num_layers) {
    const auto expected = static_cast<std::size_t>(num_layers);
    if (names.cache_k.size() != expected || names.cache_v.size() != expected ||
        names.present_k.size() != expected || names.present_v.size() != expected) {
        throw std::invalid_argument("Phi4MultimodalKvCache per-layer tensor name count mismatch");
    }
}

bool allocate_cache_pairs(std::vector<DeviceTensor>& cache_k, std::vector<DeviceTensor>& cache_v,
                          int32_t num_layers, int32_t max_length, int32_t kv_dim, DType cache_dtype,
                          cudaStream_t stream) {
    const auto count = static_cast<std::size_t>(num_layers);
    cache_k.reserve(count);
    cache_v.reserve(count);
    for (int32_t layer = 0; layer < num_layers; ++layer) {
        cache_k.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype, stream);
        if (!cache_k.back().ok())
            return false;
        cache_v.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype, stream);
        if (!cache_v.back().ok())
            return false;
    }
    return true;
}

} // namespace

Phi4MultimodalKvCache::Phi4MultimodalKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim,
                                             cudaStream_t stream, DType cache_dtype,
                                             Phi4MultimodalKvCacheNames names)
    : num_layers_(num_layers), max_length_(max_length), kv_dim_(kv_dim), cache_dtype_(cache_dtype),
      names_(std::move(names)) {
    if (num_layers <= 0 || max_length <= 0 || kv_dim <= 0)
        throw std::invalid_argument("Phi4MultimodalKvCache geometry must be positive");

    populate_default_cache_names(names_, num_layers);
    validate_cache_name_count(names_, num_layers);
    if (!allocate_cache_pairs(cache_k_, cache_v_, num_layers, max_length, kv_dim, cache_dtype,
                              stream))
        return;
    reset();
}

void Phi4MultimodalKvCache::validate_contract(TrtModule& module) const {
    validate_scalar_input(module, names_.cache_write_indices);
    validate_scalar_input(module, names_.key_value_lengths);
    if (module.has_input("attention_mask")) {
        throw std::runtime_error(
            "Phi-4 Multimodal bundle contains the removed legacy attention-mask KV contract");
    }
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        validate_cache_pair(module, names_.cache_k[index], names_.present_k[index], max_length_,
                            kv_dim_, cache_dtype_);
        validate_cache_pair(module, names_.cache_v[index], names_.present_v[index], max_length_,
                            kv_dim_, cache_dtype_);
    }
}

void Phi4MultimodalKvCache::bind_native_cache(TrtModule& module) {
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        module.bind_external(names_.cache_k[index], cache_k_[index].data());
        module.bind_external(names_.cache_v[index], cache_v_[index].data());
        if (module.device_ptr(names_.cache_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.present_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.cache_v[index]) != cache_v_[index].data() ||
            module.device_ptr(names_.present_v[index]) != cache_v_[index].data()) {
            throw std::runtime_error(
                "Phi-4 Multimodal native KV engine did not preserve cache/present aliasing");
        }
    }
}

void Phi4MultimodalKvCache::bind_to(TrtModule& module) {
    has_position_input_ = module.has_input(names_.position_id);
    validate_contract(module);
    bind_native_cache(module);
}

void Phi4MultimodalKvCache::bind_cache_inputs(TrtModule& module) {
    bind_to(module);
}

void Phi4MultimodalKvCache::write_position_input(TensorMap& inputs, int32_t seq_len) {
    if (!has_position_input_)
        return;
    pos_buf_vec_.resize(static_cast<std::size_t>(seq_len));
    for (int32_t index = 0; index < seq_len; ++index)
        pos_buf_vec_[static_cast<std::size_t>(index)] = position_ + index;
    inputs[names_.position_id] =
        Tensor{pos_buf_vec_.data(), {static_cast<int64_t>(seq_len)}, DType::kInt32};
}

void Phi4MultimodalKvCache::prepare_step(TensorMap& inputs, int32_t seq_len) {
    if (seq_len <= 0)
        throw std::invalid_argument("Phi-4 Multimodal native KV step length must be positive");
    if (seq_len > max_length_ - position_) {
        throw std::runtime_error(
            "Phi-4 Multimodal sequence exceeds the model's complete context capacity");
    }
    write_position_input(inputs, seq_len);
    cache_write_index_ = position_;
    key_value_length_ = position_ + seq_len;
    inputs[names_.cache_write_indices] = Tensor{&cache_write_index_, {1}, DType::kInt32};
    inputs[names_.key_value_lengths] = Tensor{&key_value_length_, {1}, DType::kInt32};
}

void Phi4MultimodalKvCache::validate_aliases(const std::vector<const void*>& present_k,
                                             const std::vector<const void*>& present_v) const {
    if (present_k.size() != cache_k_.size() || present_v.size() != cache_v_.size()) {
        throw std::runtime_error("Phi-4 Multimodal native KV per-layer pointer count mismatch");
    }
    for (std::size_t layer = 0; layer < cache_k_.size(); ++layer) {
        if (present_k[layer] != cache_k_[layer].data() ||
            present_v[layer] != cache_v_[layer].data()) {
            throw std::runtime_error(
                "Phi-4 Multimodal native prefill outputs must alias the KV cache");
        }
    }
}

void Phi4MultimodalKvCache::commit_prefill(const std::vector<const void*>& present_k,
                                           const std::vector<const void*>& present_v,
                                           int32_t seq_len) {
    if (seq_len <= 0 || seq_len > max_length_ - position_)
        throw std::invalid_argument("Phi-4 Multimodal native prefill exceeds capacity");
    validate_aliases(present_k, present_v);
    position_ += seq_len;
}

void Phi4MultimodalKvCache::advance(int32_t n_tokens) {
    if (n_tokens <= 0 || n_tokens > max_length_ - position_)
        throw std::runtime_error(
            "Phi-4 Multimodal sequence exceeds the model's complete context capacity");
    position_ += n_tokens;
}

void Phi4MultimodalKvCache::reset() {
    position_ = 0;
    cache_write_index_ = 0;
    key_value_length_ = 0;
}

std::size_t Phi4MultimodalKvCache::device_memory_bytes() const {
    std::size_t total = 0;
    for (const auto& tensor : cache_k_)
        total += tensor.nbytes();
    for (const auto& tensor : cache_v_)
        total += tensor.nbytes();
    return total;
}

bool Phi4MultimodalKvCache::ok() const {
    if (cache_k_.size() != static_cast<std::size_t>(num_layers_) ||
        cache_v_.size() != static_cast<std::size_t>(num_layers_))
        return false;
    for (const auto& tensor : cache_k_) {
        if (!tensor.ok())
            return false;
    }
    for (const auto& tensor : cache_v_) {
        if (!tensor.ok())
            return false;
    }
    return true;
}

} // namespace trtmc
