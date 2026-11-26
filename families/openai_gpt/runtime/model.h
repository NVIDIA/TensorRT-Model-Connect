/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/openai_gpt/runtime/kv_cache.h"
#include "families/openai_gpt/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace trtmc::openai_gpt {

struct ModelConfig {
    std::int32_t vocab_size{0};
    // Generation stops at any of these ids; the published checkpoint declares none.
    std::vector<std::int32_t> eos_token_ids;
};

// Autoregressive continuation of a text or token-ID prefix with one dual-profile engine:
// the prefill context consumes the whole prefix in one launch, the decode context produces
// one token per launch, both writing the same native KV cache.
class OpenAIGPTModel final : public internal::IModel, public internal::ITextContinuation {
  public:
    OpenAIGPTModel(std::unique_ptr<ITrtModule> prefill, std::unique_ptr<ITrtModule> decode,
                   std::unique_ptr<KvCache> cache, std::unique_ptr<Tokenizer> tokenizer,
                   ModelConfig config);

    const char* task() const noexcept override { return internal::ITextContinuation::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::TextContinuationRequest& request,
                   internal::ConfigView config) override;

  private:
    std::vector<std::int32_t> prompt_ids(const internal::TextSource& prefix,
                                         std::size_t max_tokens) const;
    void step(ITrtModule& module, const std::int32_t* tokens, std::int32_t count,
              std::vector<float>& logits);

    std::unique_ptr<ITrtModule> prefill_;
    std::unique_ptr<ITrtModule> decode_;
    std::unique_ptr<KvCache> cache_;
    std::unique_ptr<Tokenizer> tokenizer_;
    ModelConfig config_;
    std::mutex generation_;
};

} // namespace trtmc::openai_gpt
