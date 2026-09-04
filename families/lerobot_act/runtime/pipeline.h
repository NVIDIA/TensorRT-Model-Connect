/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::lerobot_act {

class Pipeline final : public IRobotControl {
  public:
    Pipeline(std::unique_ptr<ITrtModule> policy, int32_t image_height, int32_t image_width,
             int32_t image_channels, int32_t state_dim, int32_t action_dim, int32_t chunk_size,
             std::vector<float> action_min, std::vector<float> action_max);

    RobotActionChunk predict_action_chunk(const RobotObservation& observation) override;
    RobotAction act(const RobotObservation& observation) override;
    void reset() override;

  private:
    void validate_observation(const RobotObservation& observation) const;
    bool action_within_bounds(const float* action) const;

    std::unique_ptr<ITrtModule> policy_;
    int32_t image_height_{0};
    int32_t image_width_{0};
    int32_t image_channels_{0};
    int32_t state_dim_{0};
    int32_t action_dim_{0};
    int32_t chunk_size_{0};
    std::vector<float> action_min_;
    std::vector<float> action_max_;
    std::vector<float> queued_actions_;
    std::size_t next_action_{0};
};

} // namespace trtmc::lerobot_act
