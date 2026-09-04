/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

void prepare_fast_foundation_stereo_image(const float* pixels, int32_t height, int32_t width,
                                          std::vector<float>& output);

class FastFoundationStereoPipeline final : public IStereoDisparity {
  public:
    FastFoundationStereoPipeline(std::unique_ptr<ITrtModule> feature,
                                 std::unique_ptr<ITrtModule> post, std::string model_id);
    ~FastFoundationStereoPipeline() override;

    StereoDisparityResult estimate_disparity(const float* left_pixels, const float* right_pixels,
                                             int32_t height, int32_t width) override;

  private:
    void bind_post_inputs();
    void pin_input(std::vector<float>& input, bool& pinned, const char* name);
    void upload_feature_input(const char* name, const std::vector<float>& input);

    std::unique_ptr<ITrtModule> feature_;
    std::unique_ptr<ITrtModule> post_;
    std::vector<float> left_input_;
    std::vector<float> right_input_;
    std::vector<float> padded_output_;
    std::string model_id_;
    bool post_inputs_bound_{false};
    bool left_input_pinned_{false};
    bool right_input_pinned_{false};
};

} // namespace trtmc
