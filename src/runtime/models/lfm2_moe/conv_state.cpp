/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lfm2_moe/conv_state.h"

#include "trtmc/runtime/trt_module.h"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

void validate_conv_state_contract(TrtModule& module, const std::string& input,
                                  const std::string& output,
                                  const std::vector<int64_t>& expected_shape, DType dtype) {
    if (!module.has_input(input) || !module.has_output(output)) {
        throw std::runtime_error("LFM2 engine is missing convolution state pair '" + input + "'/'" +
                                 output + "'");
    }
    if (module.tensor_shape(input) != expected_shape ||
        module.tensor_shape(output) != expected_shape) {
        throw std::runtime_error("LFM2 convolution state has unexpected shape");
    }
    if (module.tensor_dtype(input) != dtype || module.tensor_dtype(output) != dtype)
        throw std::runtime_error("LFM2 convolution state dtype does not match precision");
}

void bind_conv_state_pair(TrtModule& module, const std::string& input, const std::string& output,
                          DeviceTensor& state, DeviceTensor& present) {
    module.bind_external(input, state.data());
    module.bind_external(output, present.data());
    if (module.device_ptr(input) != state.data() || module.device_ptr(output) != present.data() ||
        state.data() == present.data()) {
        throw std::runtime_error("LFM2 convolution state requires stable double buffers");
    }
}

} // namespace

Lfm2MoeConvState::Lfm2MoeConvState(int32_t num_layers, int32_t state_dim, int32_t cache_length,
                             cudaStream_t stream, DType dtype, std::string input_prefix,
                             std::string output_prefix)
    : num_layers_(num_layers), state_dim_(state_dim), cache_length_(cache_length), stream_(stream),
      dtype_(dtype), input_prefix_(std::move(input_prefix)),
      output_prefix_(std::move(output_prefix)) {
    if (num_layers_ <= 0 || state_dim_ <= 0 || cache_length_ <= 0)
        throw std::invalid_argument("Lfm2MoeConvState requires positive geometry");
    const auto count = static_cast<std::size_t>(num_layers_);
    state_.reserve(count);
    present_.reserve(count);
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        state_.emplace_back(std::vector<int64_t>{state_dim_, cache_length_}, dtype_, stream_);
        present_.emplace_back(std::vector<int64_t>{state_dim_, cache_length_}, dtype_, stream_);
    }
    if (ok())
        reset();
}

std::string Lfm2MoeConvState::input_name(int32_t layer) const {
    return input_prefix_ + "_" + std::to_string(layer);
}

std::string Lfm2MoeConvState::output_name(int32_t layer) const {
    return output_prefix_ + "_" + std::to_string(layer);
}

void Lfm2MoeConvState::bind_to(TrtModule& module) {
    const std::vector<int64_t> expected_shape{state_dim_, cache_length_};
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto input = input_name(layer);
        const auto output = output_name(layer);
        validate_conv_state_contract(module, input, output, expected_shape, dtype_);
        const auto index = static_cast<std::size_t>(layer);
        bind_conv_state_pair(module, input, output, state_[index], present_[index]);
    }
}

void Lfm2MoeConvState::advance(int32_t n_tokens) {
    if (n_tokens <= 0)
        throw std::invalid_argument("LFM2 convolution advancement must be positive");
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        if (!state_[index].copy_from(present_[index]))
            throw std::runtime_error("Failed to advance LFM2 convolution state");
    }
    position_ += n_tokens;
}

void Lfm2MoeConvState::reset() {
    position_ = 0;
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        if (cudaMemsetAsync(state_[index].data(), 0, state_[index].nbytes(), stream_) !=
                cudaSuccess ||
            cudaMemsetAsync(present_[index].data(), 0, present_[index].nbytes(), stream_) !=
                cudaSuccess) {
            throw std::runtime_error("Failed to clear LFM2 convolution state");
        }
    }
    if (cudaStreamSynchronize(stream_) != cudaSuccess)
        throw std::runtime_error("Failed to synchronize LFM2 convolution reset");
}

std::size_t Lfm2MoeConvState::device_memory_bytes() const {
    std::size_t total = 0;
    for (const auto& tensor : state_)
        total += tensor.nbytes();
    for (const auto& tensor : present_)
        total += tensor.nbytes();
    return total;
}

bool Lfm2MoeConvState::ok() const {
    const auto expected = static_cast<std::size_t>(num_layers_);
    if (state_.size() != expected || present_.size() != expected)
        return false;
    for (const auto& tensor : state_) {
        if (!tensor.ok())
            return false;
    }
    for (const auto& tensor : present_) {
        if (!tensor.ok())
            return false;
    }
    return true;
}

} // namespace trtmc
