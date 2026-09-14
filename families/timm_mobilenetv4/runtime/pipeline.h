/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/timm_mobilenetv4/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

class TimmMobilenetv4ImageClassificationPipeline final : public IImageClassification {
  public:
    explicit TimmMobilenetv4ImageClassificationPipeline(
        std::unique_ptr<ITrtModule> model, TimmMobilenetv4PreprocessConfig preprocess_config);

    ClassificationResult classify(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    TimmMobilenetv4PreprocessConfig preprocess_config_;
};

} // namespace trtmc
