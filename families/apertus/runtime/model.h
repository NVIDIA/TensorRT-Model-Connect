/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/apertus/runtime/kv_cache.h"
#include "families/apertus/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace trtmc::apertus {

struct ModelConfig {
    std::int32_t vocab_size{0};
    std::int32_t prefill_chunk_tokens{0};
    std::vector<std::int32_t> eos_token_ids;
    // Special tokens tokenizer(text) adds around a text prefix (tokenizer.json post-processor).
    std::vector<std::int32_t> prefix_ids;
    std::vector<std::int32_t> suffix_ids;
};

// Autoregressive continuation of a text or token-ID prefix with one dual-profile
// engine: the prefill context consumes the prefix in chunks, the decode context
// produces one token per launch, both writing the same native KV cache.
class ApertusModel final : public internal::IModel, public internal::ITextContinuation {
  public:
    ApertusModel(std::unique_ptr<ITrtModule> prefill, std::unique_ptr<ITrtModule> decode,
                 std::unique_ptr<KvCache> cache, std::unique_ptr<Tokenizer> tokenizer,
                 ModelConfig config);

    const char* task() const noexcept override { return internal::ITextContinuation::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::TextContinuationRequest& request,
                   internal::ConfigView config) override;

  private:
    std::vector<std::int32_t> prompt_ids(const internal::TextSource& prefix) const;
    void step(ITrtModule& module, const std::int32_t* tokens, std::int32_t count,
              std::vector<float>& logits);

    std::unique_ptr<ITrtModule> prefill_;
    std::unique_ptr<ITrtModule> decode_;
    std::unique_ptr<KvCache> cache_;
    std::unique_ptr<Tokenizer> tokenizer_;
    ModelConfig config_;
    std::mutex generation_;
};

} // namespace trtmc::apertus
