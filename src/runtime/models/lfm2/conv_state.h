/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/lfm2/inference_state.h"
#include "trtmc/runtime/device_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

class Lfm2ConvState final : public Lfm2InferenceState {
  public:
    Lfm2ConvState(int32_t num_layers, int32_t state_dim, int32_t cache_length, cudaStream_t stream,
                  DType dtype, std::string input_prefix = "conv_state",
                  std::string output_prefix = "present_conv");

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap&, int32_t = 1) override {}
    void advance(int32_t n_tokens = 1) override;

    int32_t position() const override { return position_; }
    int32_t max_length() const override { return -1; }
    int32_t num_layers() const override { return num_layers_; }
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "lfm2_short_conv"; }
    bool ok() const override;

    DeviceTensor& state(int32_t layer) { return state_.at(static_cast<std::size_t>(layer)); }
    DeviceTensor& present(int32_t layer) { return present_.at(static_cast<std::size_t>(layer)); }

  private:
    std::string input_name(int32_t layer) const;
    std::string output_name(int32_t layer) const;

    std::vector<DeviceTensor> state_;
    std::vector<DeviceTensor> present_;
    int32_t num_layers_{0};
    int32_t state_dim_{0};
    int32_t cache_length_{0};
    int32_t position_{0};
    cudaStream_t stream_{nullptr};
    DType dtype_{DType::kFloat16};
    std::string input_prefix_;
    std::string output_prefix_;
};

} // namespace trtmc
