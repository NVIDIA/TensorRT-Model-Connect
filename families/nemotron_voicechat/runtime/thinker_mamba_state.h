/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/nemotron_voicechat/runtime/thinker_inference_state.h"
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
    void bind_to(ITrtModule& module) override;
    void prepare_step(TensorMap& inputs) override;
    void advance() override;
    bool ok() const override;

    void capture_prompt_snapshot();
    void restore_prompt_snapshot();
    bool prompt_snapshot_ready() const noexcept { return prompt_snapshot_ready_; }

  private:
    std::vector<TensorSpec> specs_;
    std::vector<std::vector<DeviceTensor>> state_;
    std::vector<std::vector<DeviceTensor>> present_;
    std::vector<std::vector<DeviceTensor>> prompt_snapshot_;
    int32_t num_layers_{0};
    cudaStream_t stream_{nullptr};
    bool prompt_snapshot_ready_{false};
};

} // namespace trtmc
