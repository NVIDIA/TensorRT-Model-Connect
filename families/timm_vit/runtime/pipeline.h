/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/timm_vit/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>
#include <string>

namespace trtmc {

class ImageClassificationPipeline final : public IImageClassification {
  public:
    explicit ImageClassificationPipeline(std::unique_ptr<ITrtModule> model,
                                         TimmVitPreprocessConfig preprocess_config = {},
                                         std::string model_id_str = "");

    ClassificationResult classify(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    TimmVitPreprocessConfig preprocess_config_;
    std::string model_id_;
};

} // namespace trtmc
