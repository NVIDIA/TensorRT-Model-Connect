/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/timm_convnext/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

class TimmConvNeXtImageClassificationPipeline final : public IImageClassification {
  public:
    explicit TimmConvNeXtImageClassificationPipeline(
        std::unique_ptr<ITrtModule> model, TimmConvNeXtPreprocessConfig preprocess_config = {});

    ClassificationResult classify(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    TimmConvNeXtPreprocessConfig preprocess_config_;
};

} // namespace trtmc
