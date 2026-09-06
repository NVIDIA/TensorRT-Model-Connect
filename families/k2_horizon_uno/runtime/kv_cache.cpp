/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/k2_horizon_uno/runtime/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

constexpr int32_t kHeadDim = 128;
constexpr int32_t kMaximumBlockLength = 8;
constexpr int32_t kNumKvHeads = 8;
constexpr int32_t kNumLayers = 36;

void require_dynamic_input(ITrtModule& module, const std::string& name, DType dtype) {
    if (!module.has_input(name) || module.tensor_dtype(name) != dtype ||
        !module.input_is_dynamic(name)) {
        throw std::runtime_error("K2-Horizon-Uno KV input contract mismatch for '" + name + "'");
    }
}

void require_static_input(ITrtModule& module, const std::string& name, DType dtype,
                          const std::vector<int64_t>& shape) {
    if (!module.has_input(name) || module.tensor_dtype(name) != dtype ||
        module.tensor_shape(name) != shape || module.input_is_dynamic(name)) {
        throw std::runtime_error("K2-Horizon-Uno KV input contract mismatch for '" + name + "'");
    }
}

void validate_cache_pair(ITrtModule& module, const std::string& cache_name,
                         const std::string& present_name,
                         const std::vector<int64_t>& expected_shape) {
    if (!module.has_input(cache_name) || !module.has_output(present_name) ||
        module.tensor_shape(cache_name) != expected_shape ||
        module.tensor_shape(present_name) != expected_shape ||
        module.tensor_dtype(cache_name) != DType::kBFloat16 ||
        module.tensor_dtype(present_name) != DType::kBFloat16) {
        throw std::runtime_error(
            "K2-Horizon-Uno cache/present tensors must be aliased BF16 [1,Hkv,K,128]");
    }
}

bool all_tensors_ok(const std::vector<DeviceTensor>& tensors) {
    return std::all_of(tensors.begin(), tensors.end(),
                       [](const DeviceTensor& tensor) { return tensor.ok(); });
}

void validate_cache_names(const K2HorizonUnoKvCacheNames& names) {
    constexpr auto expected = static_cast<std::size_t>(kNumLayers);
    if (names.cache_k.size() != expected || names.cache_v.size() != expected ||
        names.present_k.size() != expected || names.present_v.size() != expected) {
        throw std::invalid_argument("K2-Horizon-Uno per-layer KV name count mismatch");
    }
}

bool allocate_cache_tensors(std::vector<DeviceTensor>& cache_k, std::vector<DeviceTensor>& cache_v,
                            const std::vector<int64_t>& shape, cudaStream_t stream) {
    cache_k.reserve(kNumLayers);
    cache_v.reserve(kNumLayers);
    for (int32_t layer = 0; layer < kNumLayers; ++layer) {
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

K2HorizonUnoCacheCursor::K2HorizonUnoCacheCursor(int32_t capacity) : capacity_(capacity) {
    if (capacity_ <= 0)
        throw std::invalid_argument("K2-Horizon-Uno KV capacity must be positive");
}

void K2HorizonUnoCacheCursor::validate_block(int32_t sequence_length) const {
    if (sequence_length <= 0 || sequence_length > kMaximumBlockLength) {
        throw std::invalid_argument("K2-Horizon-Uno block length must be in [1, 8]");
    }
    if (position_ > capacity_ - sequence_length)
        throw std::runtime_error("K2-Horizon-Uno sequence exceeds fixed KV capacity");
}

void K2HorizonUnoCacheCursor::advance(int32_t sequence_length) {
    validate_block(sequence_length);
    position_ += sequence_length;
}

void K2HorizonUnoCacheCursor::rollback(int32_t position) {
    if (position < 0 || position > position_)
        throw std::invalid_argument("K2-Horizon-Uno rollback must stay within committed KV");
    position_ = position;
}

K2HorizonUnoKvCache::K2HorizonUnoKvCache(int32_t max_length, cudaStream_t stream,
                                         K2HorizonUnoKvCacheNames names)
    : cursor_(max_length), names_(std::move(names)) {
    validate_cache_names(names_);
    const std::vector<int64_t> shape{1, kNumKvHeads, max_length, kHeadDim};
    (void)allocate_cache_tensors(cache_k_, cache_v_, shape, stream);
}

void K2HorizonUnoKvCache::validate_engine_contract(ITrtModule& module) const {
    require_dynamic_input(module, names_.position_id, DType::kInt32);
    require_static_input(module, names_.cache_write_indices, DType::kInt32, {1});
    require_static_input(module, names_.key_value_lengths, DType::kInt32, {1});
    const std::vector<int64_t> shape{1, kNumKvHeads, max_length(), kHeadDim};
    for (int32_t layer = 0; layer < kNumLayers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        validate_cache_pair(module, names_.cache_k[index], names_.present_k[index], shape);
        validate_cache_pair(module, names_.cache_v[index], names_.present_v[index], shape);
    }
}

void K2HorizonUnoKvCache::bind_cache_aliases(ITrtModule& module) {
    for (int32_t layer = 0; layer < kNumLayers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        module.bind_external(names_.cache_k[index], cache_k_[index].data());
        module.bind_external(names_.cache_v[index], cache_v_[index].data());
        if (module.device_ptr(names_.cache_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.present_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.cache_v[index]) != cache_v_[index].data() ||
            module.device_ptr(names_.present_v[index]) != cache_v_[index].data()) {
            throw std::runtime_error(
                "K2-Horizon-Uno engine did not preserve cache/present aliasing");
        }
    }
}

void K2HorizonUnoKvCache::bind_to(ITrtModule& module) {
    if (!ok())
        throw std::runtime_error("K2-Horizon-Uno KV allocation is incomplete");
    validate_engine_contract(module);
    bind_cache_aliases(module);
    bound_ = true;
}

void K2HorizonUnoKvCache::prepare_block(TensorMap& inputs, int32_t sequence_length) {
    if (!bound_)
        throw std::runtime_error("K2-Horizon-Uno KV cache must be bound before inference");
    cursor_.validate_block(sequence_length);

    position_ids_.resize(static_cast<std::size_t>(sequence_length));
    for (int32_t row = 0; row < sequence_length; ++row)
        position_ids_[static_cast<std::size_t>(row)] = cursor_.position() + row;
    cache_write_index_ = cursor_.position();
    key_value_length_ = cursor_.position() + sequence_length;
    inputs[names_.position_id] =
        Tensor{position_ids_.data(), {static_cast<int64_t>(sequence_length)}, DType::kInt32};
    inputs[names_.cache_write_indices] = Tensor{&cache_write_index_, {1}, DType::kInt32};
    inputs[names_.key_value_lengths] = Tensor{&key_value_length_, {1}, DType::kInt32};
}

void K2HorizonUnoKvCache::advance(int32_t sequence_length) {
    cursor_.advance(sequence_length);
}

void K2HorizonUnoKvCache::rollback(int32_t position) {
    cursor_.rollback(position);
}

void K2HorizonUnoKvCache::reset() {
    cursor_.reset();
    position_ids_.clear();
    cache_write_index_ = 0;
    key_value_length_ = 0;
}

bool K2HorizonUnoKvCache::ok() const {
    constexpr auto expected = static_cast<std::size_t>(kNumLayers);
    return cache_k_.size() == expected && cache_v_.size() == expected && all_tensors_ok(cache_k_) &&
           all_tensors_ok(cache_v_);
}

} // namespace trtmc
