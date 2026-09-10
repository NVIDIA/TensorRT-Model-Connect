/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/yolov10/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

class Yolov10ObjectDetectionPipeline final : public IObjectDetection {
  public:
    Yolov10ObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                   Yolov10PreprocessConfig preprocess_config,
                                   float score_threshold);

    ObjectDetectionResult detect(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    Yolov10PreprocessConfig preprocess_config_;
    float score_threshold_;
};

} // namespace trtmc
