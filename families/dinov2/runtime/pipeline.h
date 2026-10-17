/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/dinov2/runtime/image_preprocess.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

struct Dinov2RuntimeConfig {
    Dinov2PreprocessConfig preprocess;
    int32_t patch_size{14};
    int32_t hidden_size{0};
    int32_t num_register_tokens{0};
};

class Dinov2FeaturePipeline final : public internal::IModel,
                                    public internal::IImageToTokenAndPooledFeatures {
  public:
    Dinov2FeaturePipeline(std::unique_ptr<ITrtModule> model, Dinov2RuntimeConfig config);

    const char* task() const noexcept override {
        return IImageToTokenAndPooledFeatures::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImageToTokenAndPooledFeatures>(*this)};
    }
    internal::ImageTokenAndPooledFeaturesResult
    run(const internal::ImageToTokenAndPooledFeaturesRequest& request,
        internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    Dinov2RuntimeConfig config_;
    uint64_t grid_rows_{0};
    uint64_t grid_columns_{0};
    uint64_t token_count_{0};
};

} // namespace trtmc
