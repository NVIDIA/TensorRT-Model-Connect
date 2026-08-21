/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/nemotron_voicechat/thinker_inference_state.h"
#include "trtmc/runtime/device_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

class VoiceChatThinkerMambaState final : public VoiceChatThinkerInferenceState {
  public:
    struct TensorSpec {
        std::string name;
        std::vector<int64_t> shape;
        std::string output_prefix;
    };

    VoiceChatThinkerMambaState(int32_t num_layers, std::vector<TensorSpec> specs,
                               cudaStream_t stream);

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs) override;
    void advance() override;
    bool ok() const override;

  private:
    std::vector<TensorSpec> specs_;
    std::vector<std::vector<DeviceTensor>> state_;
    std::vector<std::vector<DeviceTensor>> present_;
    int32_t num_layers_{0};
    cudaStream_t stream_{nullptr};
};

} // namespace trtmc
