/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/yolov8/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

// Greedy non-maximum suppression, per class, ordered by score. Exposed so the
// suppression can be tested without standing up an engine.
std::vector<DetectionBox> suppress_yolov8_boxes(std::vector<DetectionBox> boxes,
                                                float iou_threshold, std::size_t max_detections);

class Yolov8ObjectDetectionPipeline final : public IObjectDetection {
  public:
    Yolov8ObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                  Yolov8PreprocessConfig preprocess_config, float score_threshold,
                                  float iou_threshold, std::int32_t max_detections);

    ObjectDetectionResult detect(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    Yolov8PreprocessConfig preprocess_config_;
    float score_threshold_;
    float iou_threshold_;
    std::int32_t max_detections_;
};

} // namespace trtmc
