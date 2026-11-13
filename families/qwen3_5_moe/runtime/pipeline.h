/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// RecurrentPipeline: Qwen3.5-MoE-owned hybrid recurrent text pipeline.
// Uses Qwen35MoeInferenceState for recurrent state ownership.

#include "families/qwen3_5_moe/runtime/inference_state.h"
#include "families/qwen3_5_moe/runtime/mtp_scheduler.h"
#include "families/qwen3_5_moe/runtime/paired.h"
#include "families/qwen3_5_moe/runtime/sampler.h"
#include "families/qwen3_5_moe/runtime/tokenizer.h"
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
    int32_t id_eos{0};
    std::vector<int32_t> id_eos_ids{};
    bool has_position_input{false};
    std::string chat_template_format{};
};

// Request-owned visual rows and provider MRoPE coordinates. Byte storage preserves
// the decoder precision; a row is substituted only at its matching prompt token.
struct Qwen35MoeMediaPrompt {
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
                      std::unique_ptr<Qwen35MoeInferenceState> state, RecurrentGenConfig config,
                      cudaStream_t stream, const char* name,
                      std::shared_ptr<ITokenizer> tokenizer = nullptr,
                      std::string model_id_str = "",
                      std::unique_ptr<Qwen35MoeMtpScheduler> mtp_scheduler = nullptr,
                      std::unique_ptr<Qwen35MoePairedDraft> paired_draft = nullptr);

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg = {}) override;
    int32_t default_max_new_tokens() const override { return 128; }

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids,
                                  const TextGenerationConfig& cfg);
    GenerationResult generate_media_ids(const std::vector<int32_t>& input_ids,
                                        const Qwen35MoeMediaPrompt& media,
                                        const TextGenerationConfig& cfg);

  private:
    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<Qwen35MoeInferenceState> state_;
    RecurrentGenConfig config_;
    cudaStream_t stream_;
    const char* name_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<Qwen35MoeMtpScheduler> mtp_scheduler_;
    std::unique_ptr<Qwen35MoePairedDraft> paired_draft_;
    bool media_inputs_{false};
    int32_t media_hidden_size_{0};
    DType media_dtype_{DType::kFloat16};
    std::vector<uint8_t> zero_visual_;
    const Qwen35MoeMediaPrompt* media_prompt_{nullptr};

    std::vector<int32_t> generate_from_ids(const std::vector<int32_t>& input_ids,
                                           int32_t max_new_tokens,
                                           const Qwen35MoeSamplingParams& params);

    std::vector<int32_t> generate_from_ids_speculative(const std::vector<int32_t>& input_ids,
                                                       int32_t max_new_tokens,
                                                       const Qwen35MoeSamplingParams& params,
                                                       Qwen35MoeISampler& sampler);
    std::vector<int32_t> generate_paired(const std::vector<int32_t>& input_ids,
                                         int32_t max_new_tokens,
                                         const Qwen35MoeSamplingParams& params);
    void run_step(int32_t token_id, std::vector<float>& logits,
                  std::vector<float>* hidden_state_out = nullptr);
    void* hidden_state_device_ptr_{nullptr};
    std::size_t hidden_state_numel_{0};

    using SteadyClock = std::chrono::steady_clock;
    void report_timing(SteadyClock::time_point t_prefill_start,
                       SteadyClock::time_point t_prefill_end,
                       SteadyClock::time_point t_decode_start, SteadyClock::time_point t_decode_end,
                       int prefill_tokens, int decode_steps);

    // Cached logits output metadata (resolved once, reused every step)
    void* logits_device_ptr_{nullptr};
    std::size_t logits_numel_{0};

    // Per-step profiling accumulators
    double prof_prepare_ms_{0};
    double prof_forward_ms_{0};
    double prof_logits_copy_ms_{0};
    double prof_advance_ms_{0};
    int prof_steps_{0};
};

} // namespace trtmc
