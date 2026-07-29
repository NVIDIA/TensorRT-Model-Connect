/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/mistral/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

void validate_native_scalar_input(TrtModule& module, const std::string& name) {
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kInt32 ||
        module.tensor_shape(name) != std::vector<int64_t>{1}) {
        throw std::runtime_error("Mistral native KV input '" + name + "' must be int32 [1]");
    }
}

bool valid_native_cache_shape(const std::vector<int64_t>& shape, int32_t max_length,
                              int32_t kv_dim) {
    return shape.size() == 4 && shape[0] == 1 && shape[2] == static_cast<int64_t>(max_length) &&
           shape[1] > 0 && shape[3] > 0 && shape[1] * shape[3] == kv_dim;
}

void validate_native_cache_pair(TrtModule& module, const std::string& cache_name,
                                const std::string& present_name, int32_t max_length, int32_t kv_dim,
                                DType cache_dtype) {
    if (!module.has_input(cache_name) || !module.has_output(present_name)) {
        throw std::runtime_error("Mistral native KV engine is missing cache/present pair '" +
                                 cache_name + "'/'" + present_name + "'");
    }
    const auto cache_shape = module.tensor_shape(cache_name);
    if (!valid_native_cache_shape(cache_shape, max_length, kv_dim) ||
        module.tensor_shape(present_name) != cache_shape) {
        throw std::runtime_error("Mistral native KV cache/present tensors must share static "
                                 "[1,Hkv,max_length,D] shape");
    }
    if (module.tensor_dtype(cache_name) != cache_dtype ||
        module.tensor_dtype(present_name) != cache_dtype) {
        throw std::runtime_error("Mistral native KV cache dtype does not match model precision");
    }
}

bool all_tensors_ok(const std::vector<DeviceTensor>& tensors) {
    return std::all_of(tensors.begin(), tensors.end(),
                       [](const DeviceTensor& tensor) { return tensor.ok(); });
}

MistralKvCacheNames default_cache_names(int32_t num_layers) {
    MistralKvCacheNames names;
    names.cache_k.reserve(static_cast<std::size_t>(num_layers));
    names.cache_v.reserve(static_cast<std::size_t>(num_layers));
    names.present_k.reserve(static_cast<std::size_t>(num_layers));
    names.present_v.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t layer = 0; layer < num_layers; ++layer) {
        const std::string suffix = "_" + std::to_string(layer);
        names.cache_k.push_back("cache_k" + suffix);
        names.cache_v.push_back("cache_v" + suffix);
        names.present_k.push_back("present_k" + suffix);
        names.present_v.push_back("present_v" + suffix);
    }
    return names;
}

void validate_cache_name_count(const MistralKvCacheNames& names, std::size_t expected) {
    if (names.cache_k.size() != expected || names.cache_v.size() != expected ||
        names.present_k.size() != expected || names.present_v.size() != expected) {
        throw std::invalid_argument("MistralKvCache per-layer tensor name count mismatch");
    }
}

std::pair<std::vector<DeviceTensor>, std::vector<DeviceTensor>>
allocate_cache_tensors(int32_t num_layers, int32_t max_length, int32_t kv_dim, DType cache_dtype,
                       cudaStream_t stream) {
    std::vector<DeviceTensor> cache_k;
    std::vector<DeviceTensor> cache_v;
    cache_k.reserve(static_cast<std::size_t>(num_layers));
    cache_v.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t layer = 0; layer < num_layers; ++layer) {
        cache_k.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype, stream);
        if (!cache_k.back().ok())
            break;
        cache_v.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype, stream);
        if (!cache_v.back().ok())
            break;
    }
    return {std::move(cache_k), std::move(cache_v)};
}

} // namespace

MistralKvCache::MistralKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim,
                               cudaStream_t stream, DType cache_dtype, MistralKvCacheNames names)
    : num_layers_(num_layers), max_length_(max_length), kv_dim_(kv_dim), cache_dtype_(cache_dtype),
      names_(std::move(names)) {
    if (num_layers <= 0 || max_length <= 0 || kv_dim <= 0)
        throw std::invalid_argument("MistralKvCache geometry must be positive");

    if (names_.cache_k.empty())
        names_ = default_cache_names(num_layers);
    const auto expected = static_cast<std::size_t>(num_layers);
    validate_cache_name_count(names_, expected);
    auto caches = allocate_cache_tensors(num_layers, max_length, kv_dim, cache_dtype, stream);
    cache_k_ = std::move(caches.first);
    cache_v_ = std::move(caches.second);
    reset();
}

void MistralKvCache::validate_native_kv_contract(TrtModule& module) const {
    validate_native_scalar_input(module, names_.cache_write_indices);
    validate_native_scalar_input(module, names_.key_value_lengths);
    if (!module.has_input(names_.position_id) ||
        module.tensor_dtype(names_.position_id) != DType::kInt32) {
        throw std::runtime_error("Mistral native KV engine requires an int32 position_id input");
    }
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        validate_native_cache_pair(module, names_.cache_k[index], names_.present_k[index],
                                   max_length_, kv_dim_, cache_dtype_);
        validate_native_cache_pair(module, names_.cache_v[index], names_.present_v[index],
                                   max_length_, kv_dim_, cache_dtype_);
    }
}

void MistralKvCache::bind_native_cache(TrtModule& module) {
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        module.bind_external(names_.cache_k[index], cache_k_[index].data());
        module.bind_external(names_.cache_v[index], cache_v_[index].data());
        if (module.device_ptr(names_.cache_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.present_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.cache_v[index]) != cache_v_[index].data() ||
            module.device_ptr(names_.present_v[index]) != cache_v_[index].data()) {
            throw std::runtime_error(
                "Mistral native KV engine did not preserve cache/present aliasing");
        }
    }
}

void MistralKvCache::bind_to(TrtModule& module) {
    validate_native_kv_contract(module);
    bind_native_cache(module);
}

void MistralKvCache::bind_cache_inputs(TrtModule& module) {
    validate_native_kv_contract(module);
    bind_native_cache(module);
}

void MistralKvCache::write_position_input(TensorMap& inputs, int32_t seq_len) {
    position_ids_.resize(static_cast<std::size_t>(seq_len));
    for (int32_t token = 0; token < seq_len; ++token)
        position_ids_[static_cast<std::size_t>(token)] = position_ + token;
    inputs[names_.position_id] =
        Tensor{position_ids_.data(), {static_cast<int64_t>(seq_len)}, DType::kInt32};
}

void MistralKvCache::write_native_kv_inputs(TensorMap& inputs, int32_t seq_len) {
    if (seq_len > max_length_ - position_)
        throw std::runtime_error("Mistral sequence exceeds the model's fixed KV cache capacity");
    cache_write_index_ = position_;
    key_value_length_ = position_ + seq_len;
    inputs[names_.cache_write_indices] = Tensor{&cache_write_index_, {1}, DType::kInt32};
    inputs[names_.key_value_lengths] = Tensor{&key_value_length_, {1}, DType::kInt32};
}

void MistralKvCache::prepare_step(TensorMap& inputs, int32_t seq_len) {
    if (seq_len <= 0)
        throw std::invalid_argument("Mistral native KV step length must be positive");
    write_position_input(inputs, seq_len);
    write_native_kv_inputs(inputs, seq_len);
}

void MistralKvCache::validate_native_aliases(const std::vector<const void*>& present_k,
                                             const std::vector<const void*>& present_v) const {
    if (present_k.size() != cache_k_.size() || present_v.size() != cache_v_.size())
        throw std::runtime_error("Mistral native KV per-layer pointer count mismatch");
    for (std::size_t layer = 0; layer < cache_k_.size(); ++layer) {
        if (present_k[layer] != cache_k_[layer].data() ||
            present_v[layer] != cache_v_[layer].data()) {
            throw std::runtime_error(
                "Mistral native prefill present tensors must alias the KV cache");
        }
    }
}

void MistralKvCache::append_prefill_kv(const std::vector<const void*>& present_k,
                                       const std::vector<const void*>& present_v, int32_t seq_len) {
    if (seq_len <= 0 || seq_len > max_length_ - position_)
        throw std::runtime_error("Mistral native prefill append exceeds cache capacity");
    validate_native_aliases(present_k, present_v);
    position_ += seq_len;
}

void MistralKvCache::advance(int32_t n_tokens) {
    if (n_tokens <= 0 || n_tokens > max_length_ - position_)
        throw std::runtime_error("Mistral native KV advance exceeds cache capacity");
    position_ += n_tokens;
}

void MistralKvCache::set_position(int32_t position) {
    if (position < 0 || position > max_length_)
        throw std::out_of_range("Mistral native KV position is outside cache capacity");
    position_ = position;
}

void MistralKvCache::reset() {
    position_ = 0;
    cache_write_index_ = 0;
    key_value_length_ = 0;
}

std::size_t MistralKvCache::device_memory_bytes() const {
    std::size_t total = 0;
    for (const auto& tensor : cache_k_)
        total += tensor.nbytes();
    for (const auto& tensor : cache_v_)
        total += tensor.nbytes();
    return total;
}

bool MistralKvCache::ok() const {
    const auto expected = static_cast<std::size_t>(num_layers_);
    return cache_k_.size() == expected && cache_v_.size() == expected && all_tensors_ok(cache_k_) &&
           all_tensors_ok(cache_v_);
}

} // namespace trtmc
