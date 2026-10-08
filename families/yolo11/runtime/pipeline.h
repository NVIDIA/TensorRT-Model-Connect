/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/yolo11/runtime/image_preprocess_seam.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/perception.h"
#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

// Greedy non-maximum suppression, per class, ordered by score. Exposed so the
// suppression can be tested without standing up an engine.
std::vector<DetectionBox> suppress_yolo11_boxes(std::vector<DetectionBox> boxes,
                                                float iou_threshold, std::size_t max_detections);

class Yolo11ObjectDetectionPipeline final : public internal::IModel,
                                            public internal::IImageToBoxes {
  public:
    Yolo11ObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                  Yolo11PreprocessConfig preprocess_config, float score_threshold,
                                  float iou_threshold, std::int32_t max_detections);

    const char* task() const noexcept override { return internal::IImageToBoxes::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImageToBoxes>(*this)};
    }
    internal::DetectedBoxesResult run(const internal::ImageToBoxesRequest& request,
                                      internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    Yolo11PreprocessConfig preprocess_config_;
    float score_threshold_;
    float iou_threshold_;
    std::int32_t max_detections_;
};

} // namespace trtmc
