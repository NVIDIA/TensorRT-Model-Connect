/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/embeddinggemma2/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

namespace trtmc {
class EmbeddingGemma2Pipeline final : public internal::IModel,
                                      public IEmbedding,
                                      public internal::ITextToEmbedding {
  public:
    EmbeddingGemma2Pipeline(std::unique_ptr<ITrtModule> encoder,
                            std::shared_ptr<ITokenizer> tokenizer, int32_t dimension,
                            int32_t vocab_size, int32_t capacity,
                            std::vector<int32_t> unsupported_ids, std::string model_id);
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ITextToEmbedding>(*this)};
    }
    EmbeddingResult embed(const std::string& text) override;
    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest&,
                                          internal::ConfigView) override;

  private:
    std::unique_ptr<ITrtModule> encoder_;
    std::shared_ptr<ITokenizer> tokenizer_;
    int32_t dimension_;
    int32_t vocab_size_;
    int32_t capacity_;
    std::vector<int32_t> unsupported_ids_;
    std::string model_id_;
};
} // namespace trtmc
