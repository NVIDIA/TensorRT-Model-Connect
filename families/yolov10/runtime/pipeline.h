/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/yolov10/runtime/image_preprocess_seam.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/perception.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

class Yolov10ObjectDetectionPipeline final : public internal::IModel,
                                             public internal::IImageToBoxes {
  public:
    Yolov10ObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                   Yolov10PreprocessConfig preprocess_config,
                                   float score_threshold);

    const char* task() const noexcept override { return internal::IImageToBoxes::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImageToBoxes>(*this)};
    }
    internal::DetectedBoxesResult run(const internal::ImageToBoxesRequest& request,
                                      internal::ConfigView config) override;

  private:
    internal::DetectedBoxesResult detect(const float* pixels, std::int32_t height,
                                         std::int32_t width);

    std::unique_ptr<ITrtModule> model_;
    Yolov10PreprocessConfig preprocess_config_;
    float score_threshold_;
};

} // namespace trtmc
