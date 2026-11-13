/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/dinov2_with_registers/runtime/image_preprocess.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct Dinov2WithRegistersRuntimeConfig {
    Dinov2WithRegistersPreprocessConfig preprocess;
    std::int32_t patch_size{14};
    std::int32_t grid{16};
    std::int32_t hidden_size{768};
    std::int32_t resample{3};
    std::int32_t num_register_tokens{4};
};

// One engine, both halves of one Task. The patch rows are the token features
// and the class row is the pooled vector; they come out of the same forward
// pass, which is what the contract asks for.
class Dinov2WithRegistersPipeline final : public internal::IModel,
                                          public internal::IImageToTokenAndPooledFeatures {
  public:
    Dinov2WithRegistersPipeline(std::unique_ptr<ITrtModule> engine,
                                Dinov2WithRegistersRuntimeConfig config, std::string primary_task);

    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::ImageTokenAndPooledFeaturesResult
    run(const internal::ImageToTokenAndPooledFeaturesRequest& request,
        internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    Dinov2WithRegistersRuntimeConfig config_;
    std::string primary_task_;
};

} // namespace trtmc
