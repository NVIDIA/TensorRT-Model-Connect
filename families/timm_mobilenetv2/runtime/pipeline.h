/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/timm_mobilenetv2/runtime/image_preprocess_seam.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>

namespace trtmc {

class TimmMobilenetv2ImageClassificationPipeline final : public internal::IModel,
                                                         public internal::IImageToClassScores {
  public:
    explicit TimmMobilenetv2ImageClassificationPipeline(
        std::unique_ptr<ITrtModule> model, TimmMobilenetv2PreprocessConfig preprocess_config,
        std::int32_t num_classes, std::string vocabulary_id, std::vector<std::string> labels);

    const char* task() const noexcept override { return IImageToClassScores::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImageToClassScores>(*this)};
    }
    internal::LabelScoresResult run(const internal::ImageToClassScoresRequest& request,
                                    internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    TimmMobilenetv2PreprocessConfig preprocess_config_;
    std::int32_t num_classes_;
    std::string vocabulary_id_;
    std::vector<std::string> labels_;
};

} // namespace trtmc
