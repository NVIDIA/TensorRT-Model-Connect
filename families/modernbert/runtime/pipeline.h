/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/modernbert/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc::modernbert {

class EncoderPipeline final : public internal::IModel,
                              public internal::ITextToPooledFeatures,
                              public internal::ITextToEmbedding,
                              public internal::ITextPairToRelevance,
                              public internal::ITextQueryDocumentsToRelevance {
  public:
    EncoderPipeline(std::unique_ptr<ITrtModule> encoder, std::string task,
                    std::shared_ptr<ITokenizer> tokenizer, std::int64_t vocab_size,
                    std::int64_t max_sequence_length);

    const char* task() const noexcept override { return task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::PooledFeaturesResult run(const internal::TextToPooledFeaturesRequest&,
                                       internal::ConfigView) override;
    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest&,
                                          internal::ConfigView) override;
    internal::RelevanceResult run(const internal::TextPairToRelevanceRequest&,
                                  internal::ConfigView) override;
    internal::DocumentRelevanceResult run(const internal::TextQueryDocumentsToRelevanceRequest&,
                                          internal::ConfigView) override;

  private:
    void require_task(std::string_view task, internal::ConfigView config) const;
    std::vector<std::int32_t> resolve_ids(const internal::TextSource&) const;
    std::vector<float> forward(const std::vector<std::int32_t>& ids);

    std::unique_ptr<ITrtModule> encoder_;
    std::string task_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::int64_t vocab_size_;
    std::int64_t max_sequence_length_;
    std::size_t fixed_sequence_length_{0};
    DType mask_dtype_;
    std::size_t hidden_size_;
};

} // namespace trtmc::modernbert
