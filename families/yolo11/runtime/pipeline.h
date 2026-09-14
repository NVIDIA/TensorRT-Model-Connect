/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/yolo11/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

// Greedy non-maximum suppression, per class, ordered by score. Exposed so the
// suppression can be tested without standing up an engine.
std::vector<DetectionBox> suppress_yolo11_boxes(std::vector<DetectionBox> boxes,
                                                float iou_threshold, std::size_t max_detections);

class Yolo11ObjectDetectionPipeline final : public IObjectDetection {
  public:
    Yolo11ObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                  Yolo11PreprocessConfig preprocess_config, float score_threshold,
                                  float iou_threshold, std::int32_t max_detections);

    ObjectDetectionResult detect(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    Yolo11PreprocessConfig preprocess_config_;
    float score_threshold_;
    float iou_threshold_;
    std::int32_t max_detections_;
};

} // namespace trtmc
