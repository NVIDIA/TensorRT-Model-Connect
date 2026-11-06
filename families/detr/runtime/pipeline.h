/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/detr/runtime/image_preprocess_seam.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/perception.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

class DetrObjectDetectionPipeline final : public internal::IModel,
                                         public internal::IImageToBoxes {
  public:
    explicit DetrObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                         DetrPreprocessConfig preprocess_config);

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
    DetrPreprocessConfig preprocess_config_;
};

} // namespace trtmc
