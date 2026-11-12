/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/ministral3/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::ministral3 {

struct EncoderSettings {
    std::string task;
    std::int64_t hidden_size{0};
    std::int64_t vocab_size{0};
    std::int64_t max_sequence_length{0};
    std::string query_prompt;
    std::string document_prompt;
};

// One encoder engine serves three Tasks: the checkpoint's sentence embedding
// (role prompt, mean pooling, L2 normalization), raw mean-pooled features and
// per-token features. The bundle header's task is the primary one.
class EncoderPipeline final : public internal::IModel,
                              public internal::ITextToEmbedding,
                              public internal::ITextToPooledFeatures,
                              public internal::ITextToTokenFeatures {
  public:
    EncoderPipeline(std::unique_ptr<ITrtModule> encoder,
                    std::shared_ptr<const TekkenTokenizer> tokenizer, EncoderSettings settings);

    const char* task() const noexcept override { return settings_.task.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest&,
                                          internal::ConfigView) override;
    internal::PooledFeaturesResult run(const internal::TextToPooledFeaturesRequest&,
                                       internal::ConfigView) override;
    internal::TokenFeaturesResult run(const internal::TextToTokenFeaturesRequest&,
                                      internal::ConfigView) override;

  private:
    struct Encoded {
        std::vector<std::int32_t> ids;
        std::vector<Token> tokens; // Empty when the request supplied token IDs.
    };

    Encoded resolve(const internal::TextSource& source) const;
    std::vector<float> forward(const std::vector<std::int32_t>& ids);
    std::vector<float> mean(const std::vector<float>& states, std::size_t rows) const;

    std::unique_ptr<ITrtModule> encoder_;
    std::shared_ptr<const TekkenTokenizer> tokenizer_;
    EncoderSettings settings_;
    std::size_t hidden_{0};
};

} // namespace trtmc::ministral3
