/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_omni/runtime/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

constexpr float kMaskedScore = -1.0e4F;

std::string tensor_name(const char* stem, std::int32_t layer) {
    return std::string(stem) + "_" + std::to_string(layer);
}

bool all_ok(const std::vector<DeviceTensor>& tensors) {
    return std::all_of(tensors.begin(), tensors.end(),
                       [](const DeviceTensor& tensor) { return tensor.ok(); });
}

} // namespace

Qwen3OmniKvCache::Qwen3OmniKvCache(std::int32_t num_layers, std::int32_t max_length,
                                   std::int32_t kv_dim, cudaStream_t stream, DType dtype)
    : num_layers_(num_layers), max_length_(max_length), kv_dim_(kv_dim), stream_(stream),
      dtype_(dtype), element_size_(dtype_size(dtype)) {
    if (num_layers <= 0 || max_length <= 0 || kv_dim <= 0 || element_size_ == 0)
        throw std::invalid_argument("Qwen3-Omni KV dimensions must be positive");
    cache_k_.reserve(static_cast<std::size_t>(num_layers));
    cache_v_.reserve(static_cast<std::size_t>(num_layers));
    present_k_.reserve(static_cast<std::size_t>(num_layers));
    present_v_.reserve(static_cast<std::size_t>(num_layers));
    for (std::int32_t layer = 0; layer < num_layers; ++layer) {
        cache_k_.emplace_back(std::vector<std::int64_t>{max_length, kv_dim}, dtype, stream);
        cache_v_.emplace_back(std::vector<std::int64_t>{max_length, kv_dim}, dtype, stream);
        present_k_.emplace_back(std::vector<std::int64_t>{1, kv_dim}, dtype, stream);
        present_v_.emplace_back(std::vector<std::int64_t>{1, kv_dim}, dtype, stream);
    }
    mask_.resize(static_cast<std::size_t>(max_length) + 1);
}

void Qwen3OmniKvCache::validate_cache_inputs(ITrtModule& module) const {
    const std::vector<std::int64_t> expected{max_length_, kv_dim_};
    for (std::int32_t layer = 0; layer < num_layers_; ++layer) {
        for (const char* stem : {"cache_k", "cache_v"}) {
            const std::string name = tensor_name(stem, layer);
            if (!module.has_input(name) || module.tensor_shape(name) != expected ||
                module.tensor_dtype(name) != dtype_) {
                throw std::runtime_error("Qwen3-Omni decoder has an invalid " + name + " input");
            }
        }
    }
    if (!module.has_input("position_id") || !module.has_input("attention_mask"))
        throw std::runtime_error("Qwen3-Omni decoder is missing position or mask inputs");
}

void Qwen3OmniKvCache::bind_to(ITrtModule& module) {
    validate_cache_inputs(module);
    if (module.tensor_shape("position_id") != std::vector<std::int64_t>{1} ||
        module.tensor_dtype("position_id") != DType::kInt32 ||
        module.tensor_shape("attention_mask") != std::vector<std::int64_t>{1, max_length_ + 1} ||
        module.tensor_dtype("attention_mask") != DType::kFloat32) {
        throw std::runtime_error("Qwen3-Omni decode position/mask contract is invalid");
    }
    const std::vector<std::int64_t> present_shape{1, kv_dim_};
    for (std::int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        const std::string cache_k = tensor_name("cache_k", layer);
        const std::string cache_v = tensor_name("cache_v", layer);
        const std::string present_k = tensor_name("present_k", layer);
        const std::string present_v = tensor_name("present_v", layer);
        if (!module.has_output(present_k) || !module.has_output(present_v) ||
            module.tensor_shape(present_k) != present_shape ||
            module.tensor_shape(present_v) != present_shape ||
            module.tensor_dtype(present_k) != dtype_ || module.tensor_dtype(present_v) != dtype_) {
            throw std::runtime_error("Qwen3-Omni decode present-KV contract is invalid");
        }
        module.bind_external(cache_k, cache_k_[index].data());
        module.bind_external(cache_v, cache_v_[index].data());
        module.bind_external(present_k, present_k_[index].data());
        module.bind_external(present_v, present_v_[index].data());
    }
}

void Qwen3OmniKvCache::bind_cache_inputs(ITrtModule& module) {
    validate_cache_inputs(module);
    for (std::int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        module.bind_external(tensor_name("cache_k", layer), cache_k_[index].data());
        module.bind_external(tensor_name("cache_v", layer), cache_v_[index].data());
    }
}

void Qwen3OmniKvCache::write_positions(TensorMap& inputs, std::int32_t sequence_length) {
    positions_.resize(static_cast<std::size_t>(sequence_length));
    for (std::int32_t token = 0; token < sequence_length; ++token)
        positions_[static_cast<std::size_t>(token)] = position_ + token;
    inputs["position_id"] = Tensor{positions_.data(), {sequence_length}, DType::kInt32};
}

void Qwen3OmniKvCache::write_mask(TensorMap& inputs, std::int32_t sequence_length) {
    const std::int32_t width = max_length_ + sequence_length;
    mask_.assign(static_cast<std::size_t>(sequence_length) * width, kMaskedScore);
    for (std::int32_t query = 0; query < sequence_length; ++query) {
        const auto row = static_cast<std::size_t>(query) * width;
        for (std::int32_t key = 0; key < position_; ++key)
            mask_[row + static_cast<std::size_t>(key)] = 0.0F;
        for (std::int32_t key = 0; key <= query; ++key)
            mask_[row + static_cast<std::size_t>(max_length_ + key)] = 0.0F;
    }
    inputs["attention_mask"] = Tensor{mask_.data(), {sequence_length, width}, DType::kFloat32};
}

void Qwen3OmniKvCache::prepare_step(TensorMap& inputs, std::int32_t sequence_length) {
    if (sequence_length <= 0 || position_ + sequence_length > max_length_)
        throw std::runtime_error("Qwen3-Omni sequence exceeds its fixed KV cache capacity");
    write_positions(inputs, sequence_length);
    write_mask(inputs, sequence_length);
}

void Qwen3OmniKvCache::write_prefill_kv(const std::vector<const void*>& keys,
                                        const std::vector<const void*>& values,
                                        std::int32_t sequence_length) {
    if (position_ != 0 || sequence_length <= 0 || sequence_length > max_length_ ||
        keys.size() != static_cast<std::size_t>(num_layers_) ||
        values.size() != static_cast<std::size_t>(num_layers_)) {
        throw std::runtime_error("Qwen3-Omni prefill KV write has invalid dimensions");
    }
    const auto bytes = static_cast<std::size_t>(sequence_length) * kv_dim_ * element_size_;
    for (std::int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        if (keys[index] == nullptr || values[index] == nullptr)
            throw std::runtime_error("Qwen3-Omni prefill KV output is null");
        cudaMemcpyAsync(cache_k_[index].data(), keys[index], bytes, cudaMemcpyDeviceToDevice,
                        stream_);
        cudaMemcpyAsync(cache_v_[index].data(), values[index], bytes, cudaMemcpyDeviceToDevice,
                        stream_);
    }
    position_ = sequence_length;
}

void Qwen3OmniKvCache::advance(std::int32_t tokens) {
    if (tokens != 1)
        throw std::invalid_argument("Qwen3-Omni KV advance requires one token");
    if (position_ >= max_length_)
        throw std::runtime_error("Qwen3-Omni sequence exceeds its fixed KV cache capacity");
    const auto row_bytes = static_cast<std::size_t>(kv_dim_) * element_size_;
    const auto offset = static_cast<std::size_t>(position_) * row_bytes;
    for (std::int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_k_[index].data()) + offset,
                        present_k_[index].data(), row_bytes, cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_v_[index].data()) + offset,
                        present_v_[index].data(), row_bytes, cudaMemcpyDeviceToDevice, stream_);
    }
    ++position_;
}

void Qwen3OmniKvCache::reset() {
    position_ = 0;
}

std::size_t Qwen3OmniKvCache::device_memory_bytes() const {
    std::size_t bytes = 0;
    for (const auto* tensors : {&cache_k_, &cache_v_, &present_k_, &present_v_}) {
        for (const auto& tensor : *tensors)
            bytes += tensor.nbytes();
    }
    return bytes;
}

bool Qwen3OmniKvCache::ok() const {
    const auto count = static_cast<std::size_t>(num_layers_);
    return cache_k_.size() == count && cache_v_.size() == count && present_k_.size() == count &&
           present_v_.size() == count && all_ok(cache_k_) && all_ok(cache_v_) &&
           all_ok(present_k_) && all_ok(present_v_);
}

} // namespace trtmc
