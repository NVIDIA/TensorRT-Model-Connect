/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/detr/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

class DetrObjectDetectionPipeline final : public IObjectDetection {
  public:
    DetrObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                DetrPreprocessConfig preprocess_config, float score_threshold);

    ObjectDetectionResult detect(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    DetrPreprocessConfig preprocess_config_;
    float score_threshold_;
};

} // namespace trtmc
