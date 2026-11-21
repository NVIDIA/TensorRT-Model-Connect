/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/grounding_dino/runtime/image_preprocess_seam.h"
#include "families/grounding_dino/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/perception.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct GroundingDinoRuntimeConfig {
    GroundingDinoPreprocessConfig preprocess;
    std::int32_t max_text_len{256};
    std::int32_t num_queries{900};
    std::int32_t d_model{256};
    float box_threshold{0.3F};
    float text_threshold{0.25F};
};

class GroundingDinoPipeline final : public internal::IModel, public internal::IImageTextToBoxes {
  public:
    GroundingDinoPipeline(std::unique_ptr<ITrtModule> model, std::shared_ptr<ITokenizer> tokenizer,
                          GroundingDinoRuntimeConfig config);

    const char* task() const noexcept override { return internal::IImageTextToBoxes::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::GroundedBoxesResult run(const internal::ImageTextToBoxesRequest& request,
                                      internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    std::shared_ptr<ITokenizer> tokenizer_;
    GroundingDinoRuntimeConfig config_;
};

} // namespace trtmc
