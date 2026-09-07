/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/timm_mobilenetv2/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

class TimmMobilenetv2ImageClassificationPipeline final : public IImageClassification {
  public:
    explicit TimmMobilenetv2ImageClassificationPipeline(
        std::unique_ptr<ITrtModule> model, TimmMobilenetv2PreprocessConfig preprocess_config);

    ClassificationResult classify(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    TimmMobilenetv2PreprocessConfig preprocess_config_;
};

} // namespace trtmc
