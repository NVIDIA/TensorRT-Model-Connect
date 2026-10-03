/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/rt_detr_v2/runtime/box_decode.h"
#include "families/rt_detr_v2/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

struct RtDetrRuntimeConfig {
    RtDetrPreprocessConfig preprocess;
    std::int32_t num_queries{300};
    std::int32_t num_labels{80};
    float score_threshold{0.3F};
};

class RtDetrV2ObjectDetectionPipeline final : public IObjectDetection {
  public:
    RtDetrV2ObjectDetectionPipeline(std::unique_ptr<ITrtModule> model, RtDetrRuntimeConfig config);

    ObjectDetectionResult detect(const float* pixels, std::int32_t height,
                                 std::int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    RtDetrRuntimeConfig config_;
};

} // namespace trtmc
