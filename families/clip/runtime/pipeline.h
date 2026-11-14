/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/clip/runtime/image_preprocess_seam.h"
#include "families/clip/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct ClipRuntimeConfig {
    ClipPreprocessConfig preprocess;
    std::int32_t max_text_len{77};
    std::int32_t projection_dim{512};
    float logit_scale{100.0F};
};

// One model, three tasks: the two towers can be asked for their own embedding,
// or asked to score a text against an image. Each task runs only the towers it
// needs, which is why the bundle keeps the engines apart.
class ClipPipeline final : public internal::IModel,
                           public internal::IImageToEmbedding,
                           public internal::ITextToEmbedding,
                           public internal::ITextImageToRelevance {
  public:
    ClipPipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> text,
                 std::shared_ptr<ITokenizer> tokenizer, ClipRuntimeConfig config,
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
    ClipRuntimeConfig config_;
    std::string primary_task_;
};

} // namespace trtmc
