/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

std::vector<float> qwen_last_token_pool_and_normalize(const std::vector<float>& hidden_states,
                                                      const std::vector<int32_t>& attention_mask,
                                                      int32_t batch_size, int32_t sequence_length,
                                                      int32_t hidden_size);

class QwenEmbeddingPipeline final : public IPipeline {
  public:
    QwenEmbeddingPipeline(std::unique_ptr<TrtModule> encoder, std::shared_ptr<ITokenizer> tokenizer,
                          int32_t eos_token_id, std::string model_id);

    EmbeddingResult embed(const std::string& text) override;
    EmbeddingResult encode(const std::string& text) override { return embed(text); }

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "QwenEmbeddingPipeline"; }

  private:
    std::unique_ptr<TrtModule> encoder_;
    std::shared_ptr<ITokenizer> tokenizer_;
    int32_t eos_token_id_;
    std::string model_id_;
};

} // namespace trtmc
