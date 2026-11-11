/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Laguna text generation: chunked prefill on the engine's multi-row profile,
// then one-row decode steps (CUDA graph) with host sampling.

#include "families/laguna/runtime/kv_cache.h"
#include "families/laguna/runtime/sampler.h"
#include "families/laguna/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::laguna {

struct PipelineConfig {
    std::int32_t vocab_size{0};
    std::int32_t num_layers{0};
    std::int32_t prefill_chunk_rows{0};
    std::vector<std::int32_t> eos_token_ids;
    bool has_chat_template{false};
};

class Pipeline final : public internal::IModel, public internal::ITextContinuation {
  public:
    Pipeline(std::unique_ptr<ITrtModule> prefill, std::unique_ptr<ITrtModule> decode,
             std::unique_ptr<KvCache> cache, std::unique_ptr<Tokenizer> tokenizer,
             PipelineConfig config);

    // The bundle's primary task is the semantic text_continuation Task.
    const char* task() const noexcept override { return internal::ITextContinuation::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::TextContinuationRequest& request,
                   internal::ConfigView config) override;

    // Family-local test seam: the prompt token IDs, the prefill context's
    // logits for the last prompt token, and the decode context's logits after
    // every prompt token and every generated token except the last.
    struct LogitsTrace {
        std::vector<std::int32_t> prompt_ids;
        std::vector<float> prefill_last;
        std::vector<std::vector<float>> rows;
    };
    LogitsTrace trace_logits(const std::string& prompt, const TextGenerationConfig& cfg);

  private:
    std::vector<std::int32_t> encode_prompt(const std::string& prompt,
                                            const TextGenerationConfig& cfg) const;
    TextResult generate_tokens(const std::vector<std::int32_t>& input_ids,
                               const TextGenerationConfig& cfg);
    SamplingParams sampling_params(const TextGenerationConfig& cfg) const;
    void reset();
    void prefill(const std::vector<std::int32_t>& input_ids, std::vector<float>& logits);
    void decode_step(std::int32_t token_id, std::vector<float>& logits);
    void prime_decoder(std::int32_t token_id);

    std::unique_ptr<ITrtModule> prefill_;
    std::unique_ptr<ITrtModule> decode_;
    std::unique_ptr<KvCache> cache_;
    std::unique_ptr<Tokenizer> tokenizer_;
    PipelineConfig config_;
    bool decode_bound_{false};
    std::atomic<bool> active_{false};
};

} // namespace trtmc::laguna
