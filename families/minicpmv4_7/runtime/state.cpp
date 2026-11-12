/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/minicpmv4_7/runtime/state.h"

#include <algorithm>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace trtmc::minicpmv4_7 {
namespace {

constexpr float kMaskedScore = -1.0e4F;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("MiniCPM-V 4.7 ") + what + ": " +
                                 cudaGetErrorString(status));
}

} // namespace

DecoderState::DecoderState(const StateGeometry& geometry, cudaStream_t stream)
    : geometry_(geometry), stream_(stream) {
    const auto& g = geometry_;
    if (g.capacity <= 0 || g.linear_layers <= 0 || g.full_layers <= 0 || g.conv_dim <= 0 ||
        g.conv_kernel <= 1 || g.linear_heads <= 0 || g.linear_head_dim <= 0 || g.kv_width <= 0)
        throw std::invalid_argument("MiniCPM-V 4.7 decoder state geometry is invalid");
    for (int32_t i = 0; i < g.linear_layers; ++i) {
        conv_.push_back(DeviceTensor::zeros({g.conv_dim, g.conv_kernel}, DType::kFloat32, stream));
        conv_present_.emplace_back(std::vector<int64_t>{g.conv_dim, g.conv_kernel}, DType::kFloat32,
                                   stream);
        const std::vector<int64_t> shape{g.linear_heads, g.linear_head_dim, g.linear_head_dim};
        ssm_.push_back(DeviceTensor::zeros(shape, DType::kFloat32, stream));
        ssm_present_.emplace_back(shape, DType::kFloat32, stream);
    }
    for (int32_t i = 0; i < g.full_layers; ++i) {
        cache_k_.push_back(DeviceTensor::zeros({g.capacity, g.kv_width}, g.cache_dtype, stream));
        cache_v_.push_back(DeviceTensor::zeros({g.capacity, g.kv_width}, g.cache_dtype, stream));
        present_k_.emplace_back(std::vector<int64_t>{1, g.kv_width}, g.cache_dtype, stream);
        present_v_.emplace_back(std::vector<int64_t>{1, g.kv_width}, g.cache_dtype, stream);
    }
    for (const auto* group : {&conv_, &conv_present_, &ssm_, &ssm_present_, &cache_k_, &cache_v_,
                              &present_k_, &present_v_})
        for (const auto& tensor : *group)
            if (!tensor.ok())
                throw std::runtime_error("MiniCPM-V 4.7 failed to allocate decoder state");
    mask_.assign(static_cast<std::size_t>(g.capacity) + 1, kMaskedScore);
}

void DecoderState::bind(ITrtModule& module) {
    const auto& g = geometry_;
    auto require = [&](const std::string& name, std::vector<int64_t> shape, DType dtype) {
        if (!(module.has_input(name) || module.has_output(name)) ||
            module.tensor_dtype(name) != dtype || module.tensor_shape(name) != shape)
            throw std::runtime_error("MiniCPM-V 4.7 decoder tensor disagrees with runtime.json: " +
                                     name);
    };
    for (int32_t i = 0; i < g.linear_layers; ++i) {
        const auto s = std::to_string(i);
        require("conv_state_" + s, {g.conv_dim, g.conv_kernel}, DType::kFloat32);
        require("present_conv_" + s, {g.conv_dim, g.conv_kernel}, DType::kFloat32);
        require("ssm_state_" + s, {g.linear_heads, g.linear_head_dim, g.linear_head_dim},
                DType::kFloat32);
        require("present_ssm_" + s, {g.linear_heads, g.linear_head_dim, g.linear_head_dim},
                DType::kFloat32);
        const auto u = static_cast<std::size_t>(i);
        module.bind_external("conv_state_" + s, conv_[u].data());
        module.bind_external("present_conv_" + s, conv_present_[u].data());
        module.bind_external("ssm_state_" + s, ssm_[u].data());
        module.bind_external("present_ssm_" + s, ssm_present_[u].data());
    }
    for (int32_t i = 0; i < g.full_layers; ++i) {
        const auto s = std::to_string(i);
        require("cache_k_" + s, {g.capacity, g.kv_width}, g.cache_dtype);
        require("cache_v_" + s, {g.capacity, g.kv_width}, g.cache_dtype);
        require("present_k_" + s, {1, g.kv_width}, g.cache_dtype);
        require("present_v_" + s, {1, g.kv_width}, g.cache_dtype);
        const auto u = static_cast<std::size_t>(i);
        module.bind_external("cache_k_" + s, cache_k_[u].data());
        module.bind_external("cache_v_" + s, cache_v_[u].data());
        module.bind_external("present_k_" + s, present_k_[u].data());
        module.bind_external("present_v_" + s, present_v_[u].data());
    }
    require("attention_mask", {1, static_cast<int64_t>(g.capacity) + 1}, DType::kFloat32);
}

void DecoderState::reset() {
    position_ = 0;
    for (auto* group : {&conv_, &ssm_})
        for (auto& tensor : *group)
            check(cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), stream_), "state reset");
    // Stale KV rows stay hidden by the mask until this sequence overwrites them.
    std::fill(mask_.begin(), mask_.end(), kMaskedScore);
    check(cudaStreamSynchronize(stream_), "state reset");
}

void DecoderState::prepare(TensorMap& inputs) {
    if (position_ >= geometry_.capacity)
        throw std::invalid_argument("MiniCPM-V 4.7 sequence exceeds the decoder capacity");
    position_input_ = position_;
    mask_.back() = 0.0F;
    inputs["position_id"] = Tensor{&position_input_, {1}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{mask_.data(), {1, static_cast<int64_t>(mask_.size())}, DType::kFloat32};
}

void DecoderState::advance() {
    const auto row =
        static_cast<std::size_t>(geometry_.kv_width) * dtype_size(geometry_.cache_dtype);
    const auto offset = static_cast<std::size_t>(position_) * row;
    for (std::size_t i = 0; i < cache_k_.size(); ++i) {
        check(cudaMemcpyAsync(static_cast<uint8_t*>(cache_k_[i].data()) + offset,
                              present_k_[i].data(), row, cudaMemcpyDeviceToDevice, stream_),
              "KV commit");
        check(cudaMemcpyAsync(static_cast<uint8_t*>(cache_v_[i].data()) + offset,
                              present_v_[i].data(), row, cudaMemcpyDeviceToDevice, stream_),
              "KV commit");
    }
    for (std::size_t i = 0; i < conv_.size(); ++i) {
        check(cudaMemcpyAsync(conv_[i].data(), conv_present_[i].data(), conv_[i].nbytes(),
                              cudaMemcpyDeviceToDevice, stream_),
              "conv state commit");
        check(cudaMemcpyAsync(ssm_[i].data(), ssm_present_[i].data(), ssm_[i].nbytes(),
                              cudaMemcpyDeviceToDevice, stream_),
              "recurrent state commit");
    }
    mask_[static_cast<std::size_t>(position_)] = 0.0F;
    ++position_;
}

} // namespace trtmc::minicpmv4_7
