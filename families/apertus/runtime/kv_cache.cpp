/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/apertus/runtime/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace trtmc::apertus {
namespace {

constexpr std::int32_t kActiveRowBucket = 32;

std::string layer_name(const char* stem, std::int32_t layer) {
    return std::string(stem) + "_" + std::to_string(layer);
}

void require_scalar_input(const ITrtModule& module, const std::string& name) {
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kInt32 ||
        module.tensor_shape(name) != std::vector<int64_t>{1}) {
        throw std::runtime_error("Apertus engine input '" + name + "' must be int32 [1]");
    }
}

} // namespace

KvCache::KvCache(std::int32_t num_layers, std::int32_t num_kv_heads, std::int32_t head_dim,
                 std::int32_t capacity, DType dtype, cudaStream_t stream)
    : num_layers_(num_layers), num_kv_heads_(num_kv_heads), head_dim_(head_dim),
      capacity_(capacity), dtype_(dtype), key_positions_(static_cast<std::size_t>(capacity)) {
    if (num_layers <= 0 || num_kv_heads <= 0 || head_dim <= 0 || capacity <= 0)
        throw std::invalid_argument("Apertus KV cache dimensions must be positive");
    std::iota(key_positions_.begin(), key_positions_.end(), 0);
    const std::vector<int64_t> shape{1, num_kv_heads, capacity, head_dim};
    for (std::int32_t layer = 0; layer < num_layers; ++layer) {
        keys_.emplace_back(shape, dtype, stream);
        values_.emplace_back(shape, dtype, stream);
    }
}

bool KvCache::ok() const {
    return std::all_of(keys_.begin(), keys_.end(), [](const auto& t) { return t.ok(); }) &&
           std::all_of(values_.begin(), values_.end(), [](const auto& t) { return t.ok(); });
}

void KvCache::bind(ITrtModule& module) const {
    require_scalar_input(module, "cache_write_indices");
    require_scalar_input(module, "key_value_lengths");
    if (!module.has_input("key_positions") || module.tensor_dtype("key_positions") != DType::kInt32)
        throw std::runtime_error("Apertus engine input 'key_positions' must be int32 [R]");
    const std::vector<int64_t> shape{1, num_kv_heads_, capacity_, head_dim_};
    for (std::int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        for (const auto& [stem, present, storage] :
             {std::tuple{"cache_k", "present_k", &keys_[index]},
              std::tuple{"cache_v", "present_v", &values_[index]}}) {
            const auto cache_name = layer_name(stem, layer);
            const auto present_name = layer_name(present, layer);
            if (!module.has_input(cache_name) || !module.has_output(present_name) ||
                module.tensor_shape(cache_name) != shape ||
                module.tensor_shape(present_name) != shape ||
                module.tensor_dtype(cache_name) != dtype_ ||
                module.tensor_dtype(present_name) != dtype_) {
                throw std::runtime_error(
                    "Apertus engine has an invalid KV cache tensor for layer " +
                    std::to_string(layer));
            }
            auto* data = const_cast<void*>(storage->data());
            module.bind_external(cache_name, data);
            if (module.device_ptr(cache_name) != data || module.device_ptr(present_name) != data)
                throw std::runtime_error("Apertus engine did not alias the KV cache in place");
        }
    }
}

std::int32_t KvCache::active_rows(std::int32_t valid_rows) const {
    if (valid_rows <= 0 || valid_rows > capacity_)
        throw std::invalid_argument("Apertus active KV rows must be within the cache capacity");
    const auto rounded = (static_cast<int64_t>(valid_rows) + kActiveRowBucket - 1) /
                         kActiveRowBucket * kActiveRowBucket;
    return static_cast<std::int32_t>(std::min<int64_t>(rounded, capacity_));
}

void KvCache::prepare(TensorMap& inputs, std::int32_t tokens) {
    if (tokens <= 0 || tokens > capacity_ - length_)
        throw std::invalid_argument("Apertus sequence exceeds the bundle's KV cache capacity");
    write_index_ = length_;
    valid_rows_ = length_ + tokens;
    positions_.resize(static_cast<std::size_t>(tokens));
    std::iota(positions_.begin(), positions_.end(), length_);
    inputs["position_id"] = Tensor{positions_.data(), {tokens}, DType::kInt32};
    inputs["cache_write_indices"] = Tensor{&write_index_, {1}, DType::kInt32};
    inputs["key_value_lengths"] = Tensor{&valid_rows_, {1}, DType::kInt32};
    inputs["key_positions"] =
        Tensor{key_positions_.data(), {active_rows(valid_rows_)}, DType::kInt32};
}

void KvCache::commit(std::int32_t tokens) {
    if (tokens <= 0 || length_ + tokens != valid_rows_)
        throw std::logic_error("Apertus KV cache commit does not match the prepared step");
    length_ = valid_rows_;
}

} // namespace trtmc::apertus
