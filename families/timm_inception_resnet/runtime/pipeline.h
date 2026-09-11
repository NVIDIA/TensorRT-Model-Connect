/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/timm_inception_resnet/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

class TimmInceptionResNetImageClassificationPipeline final : public IImageClassification {
  public:
    explicit TimmInceptionResNetImageClassificationPipeline(
        std::unique_ptr<ITrtModule> model,
        TimmInceptionResNetPreprocessConfig preprocess_config = {});

    ClassificationResult classify(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    TimmInceptionResNetPreprocessConfig preprocess_config_;
};

} // namespace trtmc
