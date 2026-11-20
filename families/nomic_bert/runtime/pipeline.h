/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/nomic_bert/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct NomicBertRuntimeConfig {
    std::int32_t max_sequence_length{512};
    std::int32_t hidden_size{768};
    std::int32_t pad_token_id{0};
};

// One encoder, one task. The engine returns the final rows and the pooling
// happens here, because pooling needs to know how many of the padded rows
// are real and the engine is built at a fixed length.
class NomicBertPipeline final : public internal::IModel, public internal::ITextToEmbedding {
  public:
    NomicBertPipeline(std::unique_ptr<ITrtModule> encoder, std::shared_ptr<ITokenizer> tokenizer,
                      NomicBertRuntimeConfig config, std::string primary_task);

    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest& request,
                                          internal::ConfigView config) override;

  private:
    std::vector<float> embed(std::string_view text, internal::EmbeddingRole role);

    std::unique_ptr<ITrtModule> encoder_;
    std::shared_ptr<ITokenizer> tokenizer_;
    NomicBertRuntimeConfig config_;
    std::string primary_task_;
};

} // namespace trtmc
