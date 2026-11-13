/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpmv4_6/runtime/state.h"

#include "trtmc/runtime/trt_module.h"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace trtmc::minicpmv4_6 {
namespace {

// Additive score of masked attention entries (model-local constant).
constexpr float kMaskedScore = -1.0e4F;

std::string indexed(const char* stem, std::int32_t index) {
    return std::string(stem) + "_" + std::to_string(index);
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("MiniCPM-V 4.6 ") + what + ": " +
                                 cudaGetErrorString(status));
}

} // namespace

DecoderState::DecoderState(const DecoderGeometry& geometry, cudaStream_t stream)
    : geometry_(geometry), stream_(stream) {
    if (geometry.attention_layers <= 0 || geometry.deltanet_layers <= 0 || geometry.kv_width <= 0 ||
        geometry.conv_width <= 0 || geometry.conv_kernel <= 0 || geometry.deltanet_heads <= 0 ||
        geometry.deltanet_head_dim <= 0 || geometry.capacity <= 1)
        throw std::invalid_argument("Invalid MiniCPM-V 4.6 decoder state geometry");
    for (std::int32_t i = 0; i < geometry.attention_layers; ++i) {
        cache_k_.push_back(DeviceTensor::zeros({geometry.capacity, geometry.kv_width},
                                               geometry.cache_dtype, stream));
        cache_v_.push_back(DeviceTensor::zeros({geometry.capacity, geometry.kv_width},
                                               geometry.cache_dtype, stream));
        present_k_.emplace_back(std::vector<int64_t>{1, geometry.kv_width}, geometry.cache_dtype,
                                stream);
        present_v_.emplace_back(std::vector<int64_t>{1, geometry.kv_width}, geometry.cache_dtype,
                                stream);
    }
    const std::vector<int64_t> conv_shape{geometry.conv_width, geometry.conv_kernel};
    const std::vector<int64_t> ssm_shape{geometry.deltanet_heads, geometry.deltanet_head_dim,
                                         geometry.deltanet_head_dim};
    for (std::int32_t i = 0; i < geometry.deltanet_layers; ++i) {
        conv_.emplace_back(conv_shape, DType::kFloat32, stream);
        present_conv_.emplace_back(conv_shape, DType::kFloat32, stream);
        ssm_.emplace_back(ssm_shape, DType::kFloat32, stream);
        present_ssm_.emplace_back(ssm_shape, DType::kFloat32, stream);
    }
    for (const auto* group : {&cache_k_, &cache_v_, &present_k_, &present_v_, &conv_,
                              &present_conv_, &ssm_, &present_ssm_})
        for (const auto& tensor : *group)
            if (!tensor.ok())
                throw std::runtime_error("MiniCPM-V 4.6 failed to allocate decoder state");
    reset();
}

void DecoderState::reset() {
    // Masks hide stale KV rows; the recurrent state must start from zero.
    position_ = 0;
    for (auto* group : {&conv_, &ssm_})
        for (auto& tensor : *group)
            check(cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), stream_),
                  "recurrent state reset failed");
}

void DecoderState::bind_caches(ITrtModule& module) {
    const std::vector<int64_t> cache_shape{geometry_.capacity, geometry_.kv_width};
    for (std::int32_t i = 0; i < geometry_.attention_layers; ++i) {
        for (const char* stem : {"cache_k", "cache_v"}) {
            const auto name = indexed(stem, i);
            if (module.tensor_shape(name) != cache_shape ||
                module.tensor_dtype(name) != geometry_.cache_dtype)
                throw std::runtime_error(
                    "MiniCPM-V 4.6 engine disagrees with the KV-cache contract");
        }
        module.bind_external(indexed("cache_k", i), cache_k_[static_cast<std::size_t>(i)].data());
        module.bind_external(indexed("cache_v", i), cache_v_[static_cast<std::size_t>(i)].data());
    }
}

void DecoderState::bind_recurrent(ITrtModule& module) {
    for (std::int32_t i = 0; i < geometry_.deltanet_layers; ++i) {
        const auto layer = static_cast<std::size_t>(i);
        module.bind_external(indexed("conv_state", i), conv_[layer].data());
        module.bind_external(indexed("present_conv", i), present_conv_[layer].data());
        module.bind_external(indexed("ssm_state", i), ssm_[layer].data());
        module.bind_external(indexed("present_ssm", i), present_ssm_[layer].data());
    }
}

void DecoderState::bind_decoder(ITrtModule& module) {
    bind_caches(module);
    for (std::int32_t i = 0; i < geometry_.attention_layers; ++i) {
        module.bind_external(indexed("present_k", i),
                             present_k_[static_cast<std::size_t>(i)].data());
        module.bind_external(indexed("present_v", i),
                             present_v_[static_cast<std::size_t>(i)].data());
    }
    bind_recurrent(module);
}

void DecoderState::bind_prefill(ITrtModule& module) {
    bind_caches(module);
    bind_recurrent(module);
}

void DecoderState::prepare(TensorMap& inputs, std::int32_t rows) {
    if (rows <= 0 || position_ + rows > geometry_.capacity)
        throw std::invalid_argument("MiniCPM-V 4.6 request exceeds the KV-cache capacity");
    positions_.resize(static_cast<std::size_t>(rows));
    for (std::int32_t i = 0; i < rows; ++i)
        positions_[static_cast<std::size_t>(i)] = position_ + i;
    inputs["position_id"] = Tensor{positions_.data(), {rows}, DType::kInt32};
    // Columns [0, capacity) address the cache (valid below position_), then the
    // new rows causally.
    const std::int32_t width = geometry_.capacity + rows;
    mask_.assign(static_cast<std::size_t>(rows) * width, kMaskedScore);
    for (std::int32_t i = 0; i < rows; ++i) {
        float* row = mask_.data() + static_cast<std::size_t>(i) * width;
        std::fill(row, row + position_, 0.0F);
        std::fill(row + geometry_.capacity, row + geometry_.capacity + i + 1, 0.0F);
    }
    inputs["attention_mask"] = Tensor{mask_.data(), {rows, width}, DType::kFloat32};
}

void DecoderState::commit_recurrent() {
    for (std::size_t i = 0; i < conv_.size(); ++i) {
        if (!conv_[i].copy_from(present_conv_[i]) || !ssm_[i].copy_from(present_ssm_[i]))
            throw std::runtime_error("MiniCPM-V 4.6 recurrent state copy failed");
    }
}

void DecoderState::commit_decoder() {
    const auto row_bytes =
        static_cast<std::size_t>(geometry_.kv_width) * dtype_size(geometry_.cache_dtype);
    const auto offset = static_cast<std::size_t>(position_) * row_bytes;
    for (std::size_t i = 0; i < cache_k_.size(); ++i) {
        check(cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_k_[i].data()) + offset,
                              present_k_[i].data(), row_bytes, cudaMemcpyDeviceToDevice, stream_),
              "KV append failed");
        check(cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_v_[i].data()) + offset,
                              present_v_[i].data(), row_bytes, cudaMemcpyDeviceToDevice, stream_),
              "KV append failed");
    }
    commit_recurrent();
    ++position_;
}

void DecoderState::commit_prefill(const ITrtModule& module, std::int32_t rows) {
    const auto row_bytes =
        static_cast<std::size_t>(geometry_.kv_width) * dtype_size(geometry_.cache_dtype);
    const auto offset = static_cast<std::size_t>(position_) * row_bytes;
    for (std::int32_t i = 0; i < geometry_.attention_layers; ++i) {
        const void* keys = module.device_ptr(indexed("present_k", i));
        const void* values = module.device_ptr(indexed("present_v", i));
        if (keys == nullptr || values == nullptr)
            throw std::runtime_error("MiniCPM-V 4.6 prefill engine has no present K/V output");
        const auto layer = static_cast<std::size_t>(i);
        check(cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_k_[layer].data()) + offset, keys,
                              static_cast<std::size_t>(rows) * row_bytes, cudaMemcpyDeviceToDevice,
                              stream_),
              "KV append failed");
        check(cudaMemcpyAsync(static_cast<std::uint8_t*>(cache_v_[layer].data()) + offset, values,
                              static_cast<std::size_t>(rows) * row_bytes, cudaMemcpyDeviceToDevice,
                              stream_),
              "KV append failed");
    }
    commit_recurrent();
    position_ += rows;
}

} // namespace trtmc::minicpmv4_6
