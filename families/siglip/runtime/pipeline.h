/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/siglip/runtime/image_preprocess_seam.h"
#include "families/siglip/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct SiglipRuntimeConfig {
    SiglipPreprocessConfig preprocess;
    std::int32_t max_text_len{64};
    std::int32_t embedding_dim{768};
    std::int32_t pad_token_id{1};
    std::int32_t resample{3};
    float logit_scale{117.33F};
    float logit_bias{-12.93F};
};

// One model, three tasks: either tower can be asked for its own embedding, or
// the two can be scored against each other. Each task runs only the towers it
// needs, which is why the bundle keeps the engines apart.
class SiglipPipeline final : public internal::IModel,
                             public internal::IImageToEmbedding,
                             public internal::ITextToEmbedding,
                             public internal::ITextImageToRelevance {
  public:
    SiglipPipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> text,
                   std::shared_ptr<ITokenizer> tokenizer, SiglipRuntimeConfig config,
                   std::string primary_task);

    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::SemanticEmbeddingResult run(const internal::ImageToEmbeddingRequest& request,
                                          internal::ConfigView config) override;
    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest& request,
                                          internal::ConfigView config) override;
    internal::RelevanceResult run(const internal::TextImageToRelevanceRequest& request,
                                  internal::ConfigView config) override;

  private:
    std::vector<float> embed_image(const internal::ImageView& image);
    std::vector<float> embed_text(std::string_view text);

    std::unique_ptr<ITrtModule> vision_;
    std::unique_ptr<ITrtModule> text_;
    std::shared_ptr<ITokenizer> tokenizer_;
    SiglipRuntimeConfig config_;
    std::string primary_task_;
};

} // namespace trtmc
