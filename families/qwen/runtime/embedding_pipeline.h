/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/qwen/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

std::vector<float> qwen_last_token_pool_and_normalize(const std::vector<float>& hidden_states,
                                                      const std::vector<int32_t>& attention_mask,
                                                      int32_t batch_size, int32_t sequence_length,
                                                      int32_t hidden_size);

class QwenEmbeddingPipeline final : public internal::IModel,
                                    public IEmbedding,
                                    public internal::ITextToEmbedding {
  public:
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ITextToEmbedding>(*this)};
    }
    QwenEmbeddingPipeline(std::unique_ptr<ITrtModule> encoder,
                          std::shared_ptr<ITokenizer> tokenizer, int32_t eos_token_id,
                          std::string model_id);

    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest&,
                                          internal::ConfigView) override;

    EmbeddingResult embed(const std::string& text) override;

  private:
    std::unique_ptr<ITrtModule> encoder_;
    std::shared_ptr<ITokenizer> tokenizer_;
    int32_t eos_token_id_;
    std::string model_id_;
};

} // namespace trtmc
