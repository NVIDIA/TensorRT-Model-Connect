/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// EncoderPipeline: single-pass encoder models (Eagle-VLM embedding, reranking
// and feature checkpoints) serving the semantic embedding, pooled-features and
// relevance Tasks.

#include "families/eagle_vlm/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc {

// One loaded bundle commits to exactly one primary task at build time (mode_).
// task_bindings() advertises only the Tasks that mode implements: embedding and
// pooled-features bundles bind one Task each, and a reranking bundle binds both
// relevance Tasks (the pair and the list-scoring form share one score head).
// A run() for a Task outside the bundle's mode raises UnsupportedTask instead
// of computing a wrong result for a capability the bundle was never built for.
class EncoderPipeline final : public internal::IModel,
                              public internal::ITextToEmbedding,
                              public internal::ITextToPooledFeatures,
                              public internal::ITextPairToRelevance,
                              public internal::ITextQueryDocumentsToRelevance {
  public:
    EncoderPipeline(std::unique_ptr<ITrtModule> encoder, std::string mode,
                    std::shared_ptr<ITokenizer> tokenizer, std::int64_t vocab_size,
                    std::string reranking_pooling = "last");

    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest& request,
                                          internal::ConfigView config) override;
    internal::PooledFeaturesResult run(const internal::TextToPooledFeaturesRequest& request,
                                       internal::ConfigView config) override;
    internal::RelevanceResult run(const internal::TextPairToRelevanceRequest& request,
                                  internal::ConfigView config) override;
    internal::DocumentRelevanceResult
    run(const internal::TextQueryDocumentsToRelevanceRequest& request,
        internal::ConfigView config) override;

  private:
    struct EncodedOutput {
        std::vector<float> data;
        std::vector<std::int64_t> shape;
    };

    std::vector<std::int32_t> resolve_ids(const internal::TextSource& text) const;
    EncodedOutput encode_ids(const std::vector<std::int32_t>& input_ids) const;
    EncodedOutput encode_batch(const std::vector<std::int32_t>& input_ids,
                               const std::vector<std::int32_t>& attention_mask,
                               std::size_t batch_size, std::size_t sequence_length) const;
    EncodedOutput forward_ids(const std::vector<std::int32_t>& input_ids,
                              const std::vector<std::int32_t>& attention_mask,
                              const std::vector<std::int64_t>& shape) const;
    float score_pair(const std::string& query, const std::string& document) const;
    std::vector<float> score_documents(const std::string& query,
                                       const std::vector<std::string>& documents) const;
    bool supports_batched_reranking() const;
    std::vector<std::int64_t> reranking_profile_max_shape() const;
    void require_mode(std::string_view expected) const;
    void require_relevance_mode() const;

    std::unique_ptr<ITrtModule> encoder_;
    // The bundle's single declared primary task: ITextToEmbedding::kTask,
    // ITextToPooledFeatures::kTask or ITextQueryDocumentsToRelevance::kTask.
    std::string mode_;
    std::shared_ptr<ITokenizer> tokenizer_;
    // Size of the model vocabulary: caller-supplied token ids must index it.
    std::int64_t vocab_size_;
    std::string reranking_pooling_;
};

} // namespace trtmc
