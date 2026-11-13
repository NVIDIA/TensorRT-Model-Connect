/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// RecurrentPipeline: Qwen3.8-owned hybrid recurrent text pipeline.
// Uses Qwen38InferenceState for recurrent state ownership.

#include "families/qwen3_8/runtime/external_draft.h"
#include "families/qwen3_8/runtime/inference_state.h"
#include "families/qwen3_8/runtime/mtp_scheduler.h"
#include "families/qwen3_8/runtime/prefill_runner.h"
#include "families/qwen3_8/runtime/sampler.h"
#include "families/qwen3_8/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct RecurrentGenConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    std::vector<int32_t> id_eos_ids;
    bool has_position_input{false};
    std::string chat_template_format{};
};

// Request-owned visual rows and provider MRoPE coordinates. Byte storage preserves
// the decoder precision; a row is substituted only at its matching prompt token.
struct Qwen38MediaPrompt {
    std::vector<std::array<int32_t, 3>> positions;
    std::vector<int32_t> feature_rows; // -1 for text, otherwise a row in features
    std::vector<uint8_t> features;
    DType dtype{DType::kFloat16};
    int32_t hidden_size{0};
    int32_t decode_position{0}; // next logical position, independent of KV length
};

class RecurrentPipeline final : public ITextGeneration {
  public:
    RecurrentPipeline(std::unique_ptr<ITrtModule> decoder,
                      std::unique_ptr<Qwen38InferenceState> state, RecurrentGenConfig config,
                      cudaStream_t stream, const char* name,
                      std::shared_ptr<ITokenizer> tokenizer = nullptr,
                      std::string model_id_str = "",
                      std::unique_ptr<Qwen38ISampler> sampler = nullptr,
                      std::unique_ptr<Qwen38MtpScheduler> mtp_scheduler = nullptr,
                      std::unique_ptr<Qwen38PrefillRunner> prefill_runner = nullptr,
                      std::unique_ptr<Qwen38ExternalDraft> external_draft = nullptr);

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg = {}) override;
    int32_t default_max_new_tokens() const override { return 128; }

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids,
                                  const TextGenerationConfig& cfg);

    GenerationResult generate_media_ids(const std::vector<int32_t>& input_ids,
                                        const Qwen38MediaPrompt& media,
                                        const TextGenerationConfig& cfg);

    static int32_t argmax(const std::vector<float>& logits);

  private:
    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<Qwen38InferenceState> state_;
    RecurrentGenConfig config_;
    cudaStream_t stream_;
    const char* name_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<Qwen38ISampler> sampler_;
    // Requested MTP drafts use the same target sampler, filters, history
    // and per-request RNG as ordinary decoding.
    std::unique_ptr<Qwen38MtpScheduler> mtp_scheduler_;
    // Present when the bundle ships prefill_engine.plan: prompts are consumed
    // chunk_len() tokens per engine call instead of one decoder step per token.
    std::unique_ptr<Qwen38PrefillRunner> prefill_runner_;
    std::unique_ptr<Qwen38ExternalDraft> external_draft_;
    bool media_inputs_{false};
    int32_t media_hidden_size_{0};
    DType media_dtype_{DType::kFloat16};
    std::vector<uint8_t> zero_visual_;
    const Qwen38MediaPrompt* media_prompt_{nullptr}; // scoped to generate_media_ids

    std::vector<int32_t> generate_external_draft(const std::vector<int32_t>& input_ids,
                                                 int32_t max_new_tokens,
                                                 const Qwen38SamplingParams& params);

    std::vector<int32_t> generate_from_ids(const std::vector<int32_t>& input_ids,
                                           int32_t max_new_tokens,
                                           const Qwen38SamplingParams& params);

    std::vector<int32_t> generate_from_ids_speculative(const std::vector<int32_t>& input_ids,
                                                       int32_t max_new_tokens,
                                                       const Qwen38SamplingParams& params,
                                                       Qwen38ISampler& sampler);

    // Feed prompt tokens [0, input_ids.size()) through the multi-token prefill
    // engine as far as the KV cache allows; returns how many were consumed
    // (always the whole prompt unless it outgrows the cache). `logits` holds
    // the last consumed token's logits. With a non-null `hidden`, it receives
    // that token's hidden state and every consumed token at position >= 1 is
    // also passed to the MTP scheduler's warm-up draft().
    std::size_t prefill_chunks(const std::vector<int32_t>& input_ids, std::vector<float>& logits,
                               std::vector<float>* hidden);

    // hidden_state_out, when non-null, is filled with the decoder's
    // hidden_state output (D2H copy) -- used by the speculative decode loop
    // to feed Qwen38MtpScheduler::draft(). Skipped (no extra D2H copy) for
    // the plain decode loop.
    void run_step(int32_t token_id, std::vector<float>& logits,
                  std::vector<float>* hidden_state_out = nullptr);

    using SteadyClock = std::chrono::steady_clock;
    void report_timing(SteadyClock::time_point t_prefill_start,
                       SteadyClock::time_point t_prefill_end,
                       SteadyClock::time_point t_decode_start, SteadyClock::time_point t_decode_end,
                       int prefill_tokens, int decode_steps);

    // Cached logits output metadata (resolved once, reused every step)
    void* logits_device_ptr_{nullptr};
    std::size_t logits_numel_{0};
    // Cached hidden_state output metadata, resolved once on first use.
    void* hidden_state_device_ptr_{nullptr};
    std::size_t hidden_state_numel_{0};

    // Per-step profiling accumulators
    double prof_prepare_ms_{0};
    double prof_forward_ms_{0};
    double prof_logits_copy_ms_{0};
    double prof_advance_ms_{0};
    int prof_steps_{0};
};

} // namespace trtmc
