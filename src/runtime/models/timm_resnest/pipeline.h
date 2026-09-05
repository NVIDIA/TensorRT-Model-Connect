/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/timm_resnest/image_preprocess_seam.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>

namespace trtmc {

class TimmResnestImageClassificationPipeline final : public IPipeline {
  public:
    explicit TimmResnestImageClassificationPipeline(
        std::unique_ptr<TrtModule> model, TimmResnestPreprocessConfig preprocess_config = {},
        std::string model_id_str = "");

    ClassificationResult classify(const float* pixels, int32_t height, int32_t width) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "TimmResnestImageClassificationPipeline"; }

  private:
    std::unique_ptr<TrtModule> model_;
    TimmResnestPreprocessConfig preprocess_config_;
    std::string model_id_;
};

} // namespace trtmc
