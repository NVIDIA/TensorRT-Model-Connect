/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/timm_hrnet/image_preprocess_seam.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>

namespace trtmc {

class TimmHrnetImageClassificationPipeline final : public IPipeline {
  public:
    explicit TimmHrnetImageClassificationPipeline(std::unique_ptr<TrtModule> model,
                                                  TimmHrnetPreprocessConfig preprocess_config = {},
                                                  std::string model_id_str = "");

    ClassificationResult classify(const float* pixels, int32_t height, int32_t width) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "TimmHrnetImageClassificationPipeline"; }

  private:
    std::unique_ptr<TrtModule> model_;
    TimmHrnetPreprocessConfig preprocess_config_;
    std::string model_id_;
};

} // namespace trtmc
