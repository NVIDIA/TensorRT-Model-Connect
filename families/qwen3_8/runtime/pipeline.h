/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// RecurrentPipeline: Qwen3.8-owned hybrid recurrent text pipeline.
// Uses Qwen38InferenceState for recurrent state ownership.

#include "families/qwen3_8/runtime/inference_state.h"
#include "families/qwen3_8/runtime/mtp_scheduler.h"
#include "families/qwen3_8/runtime/sampler.h"
#include "families/qwen3_8/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

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

class RecurrentPipeline final : public ITextGeneration {
  public:
    RecurrentPipeline(std::unique_ptr<ITrtModule> decoder,
                      std::unique_ptr<Qwen38InferenceState> state, RecurrentGenConfig config,
                      cudaStream_t stream, const char* name,
                      std::shared_ptr<ITokenizer> tokenizer = nullptr,
                      std::string model_id_str = "",
                      std::unique_ptr<Qwen38ISampler> sampler = nullptr,
                      std::unique_ptr<Qwen38MtpScheduler> mtp_scheduler = nullptr);

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg = {}) override;
    int32_t default_max_new_tokens() const override { return 128; }

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids,
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
    // Present only for checkpoints that ship mtp.* weights. When set,
    // generate_from_ids() uses the accept/reject speculative decode loop
    // instead of the plain single-token loop -- always greedy (no sampler,
    // no repetition penalty): speculative decoding's accept/reject
    // correctness depends on the verify step's argmax being exactly
    // reproducible by a plain single-token re-run, which only holds for
    // deterministic greedy decoding.
    std::unique_ptr<Qwen38MtpScheduler> mtp_scheduler_;

    std::vector<int32_t> generate_from_ids(const std::vector<int32_t>& input_ids,
                                           int32_t max_new_tokens,
                                           const Qwen38SamplingParams& params);

    std::vector<int32_t> generate_from_ids_speculative(const std::vector<int32_t>& input_ids,
                                                       int32_t max_new_tokens);

    bool is_eos(int32_t token) const;

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
