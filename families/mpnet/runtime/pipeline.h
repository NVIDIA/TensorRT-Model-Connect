/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// EncoderPipeline: single-pass encoder models (MPNet) serving the
// pooled-features and embedding Tasks, owned by mpnet and duplicated by
// its byte-identical sibling families.

#include "families/mpnet/runtime/tokenizer.h"
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

// One loaded bundle commits to exactly one of the two Tasks below at build time
// (mode_, recorded as the bundle's primary task). task_bindings() advertises
// only that Task; the other run() override rejects a mismatched call with
// UnsupportedTask instead of silently computing a wrong result for a capability
// this bundle was never built for. There is deliberately no relevance Task: the
// engine returns hidden states, not a trained relevance score.
class EncoderPipeline final : public internal::IModel,
                              public internal::ITextToEmbedding,
                              public internal::ITextToPooledFeatures {
  public:
    EncoderPipeline(std::unique_ptr<ITrtModule> encoder, std::string mode,
                    std::shared_ptr<ITokenizer> tokenizer, std::int64_t vocab_size);

    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest& request,
                                          internal::ConfigView config) override;
    internal::PooledFeaturesResult run(const internal::TextToPooledFeaturesRequest& request,
                                       internal::ConfigView config) override;

  private:
    std::vector<std::int32_t> resolve_ids(const internal::TextSource& text) const;
    std::pair<std::vector<float>, std::int32_t>
    run_encoder(const std::vector<std::int32_t>& input_ids) const;
    void require_mode(std::string_view expected) const;

    std::unique_ptr<ITrtModule> encoder_;
    // ITextToEmbedding::kTask or ITextToPooledFeatures::kTask: the bundle's
    // single declared primary task.
    std::string mode_;
    std::shared_ptr<ITokenizer> tokenizer_;
    // Size of the model vocabulary: caller-supplied token ids must index it.
    std::int64_t vocab_size_;
};

} // namespace trtmc
