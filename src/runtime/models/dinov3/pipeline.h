/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/dinov3/image_preprocess.h"
#include "trtmc/image_features.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class Dinov3ImageFeaturePipeline final : public IPipeline, public IImageFeatureExtractor {
  public:
    explicit Dinov3ImageFeaturePipeline(std::unique_ptr<TrtModule> model,
                                        Dinov3PreprocessConfig preprocess_config = {},
                                        std::string model_id = "");

    ImageFeaturesResult extract_image_features(const float* pixels, int32_t height,
                                               int32_t width) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "Dinov3ImageFeaturePipeline"; }

  private:
    std::unique_ptr<TrtModule> model_;
    Dinov3PreprocessConfig preprocess_config_;
    std::string model_id_;
    std::size_t hidden_count_{0};
    std::size_t pooler_count_{0};
    std::vector<int64_t> hidden_shape_;
    std::vector<int64_t> pooler_shape_;
};

} // namespace trtmc
