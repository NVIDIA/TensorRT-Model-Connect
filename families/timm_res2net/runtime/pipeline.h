/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/timm_res2net/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

class TimmRes2NetImageClassificationPipeline final : public IImageClassification {
  public:
    explicit TimmRes2NetImageClassificationPipeline(
        std::unique_ptr<ITrtModule> model, TimmRes2NetPreprocessConfig preprocess_config = {});

    ClassificationResult classify(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    TimmRes2NetPreprocessConfig preprocess_config_;
};

} // namespace trtmc
