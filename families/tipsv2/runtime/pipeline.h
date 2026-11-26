/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/tipsv2/runtime/image_preprocess.h"
#include "families/tipsv2/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::tipsv2 {

struct RuntimeConfig {
    PreprocessConfig preprocess;
    std::int32_t max_text_len{64};
    std::int32_t embedding_dim{1152};
    std::int32_t pad_token_id{0};
    // The rows of the text engine's embedding table: no token id may reach it out of range.
    std::int32_t vocab_size{32000};
    std::int32_t resample{2};
    // The learned contrastive temperature. The reference returns it beside the embeddings and
    // leaves the division to the caller.
    float temperature{0.004638021811842918F};
};

// One model, three tasks: either tower can be asked for its own embedding, or the two can be
// scored against each other. Each task runs only the towers it needs, which is why the bundle
// keeps the engines apart.
class Pipeline final : public internal::IModel,
                       public internal::IImageToEmbedding,
                       public internal::ITextToEmbedding,
                       public internal::ITextImageToRelevance {
  public:
    Pipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> text,
             std::shared_ptr<Tokenizer> tokenizer, RuntimeConfig config, std::string primary_task);

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
    // One per tower. `ITrtModule::forward` returns tensors that point into the module's own
    // staging buffer, so the lock has to cover the copy out of it as well as the call: without
    // it two overlapping requests on one model would silently read each other's embedding.
    std::mutex vision_mutex_;
    std::mutex text_mutex_;
    std::shared_ptr<Tokenizer> tokenizer_;
    RuntimeConfig config_;
    std::string primary_task_;
};

} // namespace trtmc::tipsv2
