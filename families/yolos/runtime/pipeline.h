/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/yolos/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

class YolosObjectDetectionPipeline final : public IObjectDetection {
  public:
    YolosObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                 YolosPreprocessConfig preprocess_config);

    ObjectDetectionResult detect(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    YolosPreprocessConfig preprocess_config_;
};

} // namespace trtmc
