/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/action.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::lerobot_act {

// One ACT policy engine serves both Tasks: the stateless chunk Task is the
// bundle's primary Task, and the action queue plays the same chunks back one
// step at a time.
class Pipeline final : public internal::IModel,
                       public internal::IImageStateToActionChunk,
                       public internal::IImageStateActionQueue {
  public:
    Pipeline(std::unique_ptr<ITrtModule> policy, int32_t image_height, int32_t image_width,
             int32_t image_channels, int32_t state_dim, int32_t action_dim, int32_t chunk_size,
             std::vector<float> action_min, std::vector<float> action_max);

    const char* task() const noexcept override {
        return internal::IImageStateToActionChunk::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::ImageStateActionChunkResult
    run(const internal::ImageStateToActionChunkRequest& request,
        internal::ConfigView config) override;
    std::unique_ptr<internal::IImageStateActionSession>
    create_action_session(internal::ConfigView config) override;

  private:
    class ActionQueue;

    internal::ImageStateActionChunkResult
    predict(const internal::ImageStateObservation& observation);
    const float* image_pixels(const internal::ImageView& image);
    void validate_shape(const internal::ImageStateObservation& observation) const;
    void validate_state(Span<const float> state) const;
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
    std::vector<float> converted_pixels_; // UInt8 observations, divided by 255.
};

} // namespace trtmc::lerobot_act
