/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

void prepare_fast_foundation_stereo_image(const float* pixels, int32_t height, int32_t width,
                                          std::vector<float>& output);

class FastFoundationStereoPipeline final : public IPipeline {
  public:
    FastFoundationStereoPipeline(std::unique_ptr<ITrtModule> feature,
                                 std::unique_ptr<ITrtModule> post, std::string model_id);

    StereoDisparityResult estimate_disparity(const float* left_pixels, const float* right_pixels,
                                             int32_t height, int32_t width) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "FastFoundationStereoPipeline"; }

  private:
    void bind_post_inputs();

    std::unique_ptr<ITrtModule> feature_;
    std::unique_ptr<ITrtModule> post_;
    std::vector<float> left_input_;
    std::vector<float> right_input_;
    std::vector<float> padded_output_;
    std::string model_id_;
    bool post_inputs_bound_{false};
};

} // namespace trtmc
