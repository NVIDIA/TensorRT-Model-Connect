/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// EncoderPipeline: single-pass encoder models (ALBERT embedding, encoding, reranking).
// Implements IModel with three semantic Task interfaces:
//   text_to_token_features  (was: encoding)
//   text_to_embedding       (was: embedding)
//   text_pair_to_relevance  (was: reranking)

#include "families/albert/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class EncoderPipeline final : public trtmc::internal::IModel,
                              public trtmc::internal::ITextToPooledFeatures,
                              public trtmc::internal::ITextToTokenFeatures,
                              public trtmc::internal::ITextToEmbedding,
                              public trtmc::internal::ITextPairToRelevance {
  public:
    EncoderPipeline(std::unique_ptr<ITrtModule> encoder, std::string primary_task,
                    std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "");

    // ITask (via IModel)
    const char* task() const noexcept override;

    // IModel
    std::vector<trtmc::internal::TaskInstance> task_bindings() override;

    // ITextToPooledFeatures — preserves legacy encoding CLS output
    // Returns single [hidden_size] pooled vector (CLS token) without normalization.
    trtmc::internal::PooledFeaturesResult
    run(const trtmc::internal::TextToPooledFeaturesRequest& request,
        trtmc::internal::ConfigView config) override;

    // ITextToTokenFeatures — was: encoding
    // Returns per-token hidden states for the full tokenized sequence.
    // The result matrix is [actual_seq_len, hidden_size]; row 0 is the CLS token.
    trtmc::internal::TokenFeaturesResult
    run(const trtmc::internal::TextToTokenFeaturesRequest& request,
        trtmc::internal::ConfigView config) override;

    // ITextToEmbedding — was: embedding
    // Mean-pools all token hidden states then L2-normalises the result.
    trtmc::internal::SemanticEmbeddingResult
    run(const trtmc::internal::TextToEmbeddingRequest& request,
        trtmc::internal::ConfigView config) override;

    // ITextPairToRelevance — was: reranking
    // Concatenates "question:<query>   passage:<document>", returns first output scalar.
    trtmc::internal::RelevanceResult run(const trtmc::internal::TextPairToRelevanceRequest& request,
                                         trtmc::internal::ConfigView config) override;

    // Token-ID-based encoding helper (for unit tests and internal callers).
    std::vector<float> encode_ids(const std::vector<int32_t>& input_ids);

  private:
    std::unique_ptr<ITrtModule> encoder_;
    std::string primary_task_; // semantic task ID stored in the bundle header
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

} // namespace trtmc
