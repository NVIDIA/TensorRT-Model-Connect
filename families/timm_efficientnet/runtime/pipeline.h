/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/timm_efficientnet/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

class TimmEfficientnetImageClassificationPipeline final : public IImageClassification {
  public:
    explicit TimmEfficientnetImageClassificationPipeline(
        std::unique_ptr<ITrtModule> model, TimmEfficientnetPreprocessConfig preprocess_config);

    ClassificationResult classify(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    TimmEfficientnetPreprocessConfig preprocess_config_;
};

} // namespace trtmc
