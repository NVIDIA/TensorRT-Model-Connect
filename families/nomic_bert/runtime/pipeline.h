/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/nomic_bert/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

namespace trtmc::nomic_bert {

class Pipeline final : public internal::IModel, public internal::ITextToEmbedding {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, std::unique_ptr<ITokenizer> tokenizer,
             int32_t length, int32_t vocab_size, std::string embedding_space);
    const char* task() const noexcept override { return "text_to_embedding"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ITextToEmbedding>(*this)};
    }
    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest&,
                                          internal::ConfigView) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    std::unique_ptr<ITokenizer> tokenizer_;
    int32_t length_;
    int32_t vocab_size_;
    std::string embedding_space_;
};

} // namespace trtmc::nomic_bert
