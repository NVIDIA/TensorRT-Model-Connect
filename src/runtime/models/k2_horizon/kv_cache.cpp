/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/k2_horizon/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

constexpr int32_t kHeadDim = 128;

void validate_int32_scalar_input(TrtModule& module, const std::string& name) {
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kInt32 ||
        module.tensor_shape(name) != std::vector<int64_t>{1}) {
        throw std::runtime_error("K2-Horizon native KV input '" + name + "' must be int32 [1]");
    }
}

void validate_cache_pair(TrtModule& module, const std::string& cache_name,
                         const std::string& present_name,
                         const std::vector<int64_t>& expected_shape) {
    if (!module.has_input(cache_name) || !module.has_output(present_name)) {
        throw std::runtime_error("K2-Horizon native KV engine is missing cache/present pair '" +
                                 cache_name + "'/'" + present_name + "'");
    }
    if (module.tensor_shape(cache_name) != expected_shape ||
        module.tensor_shape(present_name) != expected_shape) {
        throw std::runtime_error("K2-Horizon native KV cache/present tensors must use static "
                                 "[1,Hkv,max_length,128] shapes");
    }
    if (module.tensor_dtype(cache_name) != DType::kBFloat16 ||
        module.tensor_dtype(present_name) != DType::kBFloat16) {
        throw std::runtime_error("K2-Horizon native KV cache/present tensors must be BF16");
    }
}

bool all_tensors_ok(const std::vector<DeviceTensor>& tensors) {
    for (const auto& tensor : tensors) {
        if (!tensor.ok())
            return false;
    }
    return true;
}

bool name_counts_match(const K2HorizonKvCacheNames& names, std::size_t expected) {
    return names.cache_k.size() == expected && names.cache_v.size() == expected &&
           names.present_k.size() == expected && names.present_v.size() == expected;
}

void validate_cache_allocation_size(int32_t num_layers, int32_t num_kv_heads, int32_t max_length) {
    constexpr std::size_t element_bytes = 2;
    constexpr std::size_t buffers_per_layer = 2;
    const auto maximum = std::numeric_limits<std::size_t>::max();
    const auto heads = static_cast<std::size_t>(num_kv_heads);
    const auto rows = static_cast<std::size_t>(max_length);
    if (heads > maximum / rows / static_cast<std::size_t>(kHeadDim))
        throw std::overflow_error("K2-Horizon KV tensor element count overflows size_t");
    const auto elements = heads * rows * static_cast<std::size_t>(kHeadDim);
    if (elements > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) ||
        elements > maximum / element_bytes / buffers_per_layer) {
        throw std::overflow_error("K2-Horizon KV tensor byte count overflows");
    }
    const auto bytes_per_layer = elements * element_bytes * buffers_per_layer;
    if (static_cast<std::size_t>(num_layers) > maximum / bytes_per_layer)
        throw std::overflow_error("K2-Horizon total KV cache byte count overflows");
}

bool allocate_cache_tensors(std::vector<DeviceTensor>& cache_k, std::vector<DeviceTensor>& cache_v,
                            const std::vector<int64_t>& shape, int32_t num_layers,
                            cudaStream_t stream) {
    cache_k.reserve(static_cast<std::size_t>(num_layers));
    cache_v.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t layer = 0; layer < num_layers; ++layer) {
        cache_k.emplace_back(shape, DType::kBFloat16, stream);
        if (!cache_k.back().ok())
            return false;
        cache_v.emplace_back(shape, DType::kBFloat16, stream);
        if (!cache_v.back().ok())
            return false;
    }
    return true;
}

} // namespace

K2HorizonKvCache::K2HorizonKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim,
                                   cudaStream_t stream, K2HorizonKvCacheNames names)
    : num_layers_(num_layers), max_length_(max_length), names_(std::move(names)) {
    if (num_layers_ <= 0)
        throw std::invalid_argument("K2-Horizon KV layer count must be positive");
    if (max_length_ <= 0)
        throw std::invalid_argument("K2-Horizon KV capacity must be positive");
    if (kv_dim <= 0 || kv_dim % kHeadDim != 0) {
        throw std::invalid_argument(
            "K2-Horizon KV dimension must be a positive multiple of head_dim=128");
    }
    num_kv_heads_ = kv_dim / kHeadDim;
    validate_cache_allocation_size(num_layers_, num_kv_heads_, max_length_);

    const auto expected_names = static_cast<std::size_t>(num_layers_);
    if (!name_counts_match(names_, expected_names)) {
        throw std::invalid_argument("K2-Horizon KV per-layer tensor name count mismatch");
    }

    const std::vector<int64_t> cache_shape{1, num_kv_heads_, max_length_, kHeadDim};
    if (!allocate_cache_tensors(cache_k_, cache_v_, cache_shape, num_layers_, stream))
        return;
    reset();
}

void K2HorizonKvCache::validate_engine_contract(TrtModule& module) const {
    validate_int32_scalar_input(module, names_.position_id);
    validate_int32_scalar_input(module, names_.cache_write_indices);
    validate_int32_scalar_input(module, names_.key_value_lengths);

    const std::vector<int64_t> expected_shape{1, num_kv_heads_, max_length_, kHeadDim};
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        validate_cache_pair(module, names_.cache_k[index], names_.present_k[index], expected_shape);
        validate_cache_pair(module, names_.cache_v[index], names_.present_v[index], expected_shape);
    }
}

void K2HorizonKvCache::bind_cache_aliases(TrtModule& module) {
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        module.bind_external(names_.cache_k[index], cache_k_[index].data());
        module.bind_external(names_.cache_v[index], cache_v_[index].data());
        if (module.device_ptr(names_.cache_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.present_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.cache_v[index]) != cache_v_[index].data() ||
            module.device_ptr(names_.present_v[index]) != cache_v_[index].data()) {
            throw std::runtime_error(
                "K2-Horizon native KV engine did not preserve cache/present aliasing");
        }
    }
}

void K2HorizonKvCache::bind_to(TrtModule& module) {
    if (!ok())
        throw std::runtime_error("K2-Horizon native KV cache allocation is incomplete");
    validate_engine_contract(module);
    bind_cache_aliases(module);
    bound_ = true;
}

void K2HorizonKvCache::prepare_step(TensorMap& inputs, int32_t seq_len) {
    if (!bound_)
        throw std::runtime_error("K2-Horizon native KV cache must be bound before inference");
    if (seq_len != 1)
        throw std::invalid_argument("K2-Horizon native KV supports exactly one token per step");
    if (position_ >= max_length_)
        throw std::runtime_error("K2-Horizon sequence exceeds the fixed KV cache capacity");

    position_id_ = position_;
    cache_write_index_ = position_;
    key_value_length_ = position_ + 1;
    inputs[names_.position_id] = Tensor{&position_id_, {1}, DType::kInt32};
    inputs[names_.cache_write_indices] = Tensor{&cache_write_index_, {1}, DType::kInt32};
    inputs[names_.key_value_lengths] = Tensor{&key_value_length_, {1}, DType::kInt32};
}

void K2HorizonKvCache::advance(int32_t n_tokens) {
    if (n_tokens != 1)
        throw std::invalid_argument("K2-Horizon native KV supports exactly one-token advances");
    if (position_ >= max_length_)
        throw std::runtime_error("K2-Horizon sequence exceeds the fixed KV cache capacity");
    ++position_;
}

void K2HorizonKvCache::reset() {
    position_ = 0;
    position_id_ = 0;
    cache_write_index_ = 0;
    key_value_length_ = 0;
}

std::size_t K2HorizonKvCache::device_memory_bytes() const {
    std::size_t total = 0;
    for (const auto& tensor : cache_k_)
        total += tensor.nbytes();
    for (const auto& tensor : cache_v_)
        total += tensor.nbytes();
    return total;
}

bool K2HorizonKvCache::ok() const {
    const auto expected_layers = static_cast<std::size_t>(num_layers_);
    return cache_k_.size() == expected_layers && cache_v_.size() == expected_layers &&
           all_tensors_ok(cache_k_) && all_tensors_ok(cache_v_);
}

} // namespace trtmc
