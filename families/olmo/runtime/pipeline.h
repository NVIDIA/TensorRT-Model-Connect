/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Model-owned decoder text pipeline.
//
// Composes: ITrtModule (decoder) + OlmoKvCache + ITokenizer for this runtime
// plugin. Architecture-specific behavior remains in this model directory and
// in the TRT engine emitted by the matching family builder.

#include "families/olmo/runtime/inference_state.h"
#include "families/olmo/runtime/sampler.h"
#include "families/olmo/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class OlmoKvCache;

struct OlmoTextGenConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
    std::string chat_template_format{};
    std::string token_id_name{"token_id"};
    std::string logits_output_name{"logits"};
    // Required family-owned batched prefill contract.
    std::string present_k_pattern{"present_k_{i}"};
    std::string present_v_pattern{"present_v_{i}"};
    int32_t prefill_max_length{0};
    int32_t num_layers{0};
};

class OlmoTextGenerationPipeline final : public ITextGeneration {
  public:
    OlmoTextGenerationPipeline(std::unique_ptr<ITrtModule> decoder,
                               std::unique_ptr<OlmoInferenceState> state, OlmoTextGenConfig config,
                               std::shared_ptr<ITokenizer> tokenizer,
                               std::unique_ptr<ITrtModule> prefill,
                               std::shared_ptr<void> distributed_owner = nullptr);

    // Public API: takes raw text, returns typed result.
    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg = {}) override;
    int32_t default_max_new_tokens() const override { return 128; }

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids,
                                  const TextGenerationConfig& cfg);

  private:
    // Kept before TRT modules so TP communicators outlive contexts/engines.
    std::shared_ptr<void> distributed_owner_;
    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<ITrtModule> prefill_;
    std::unique_ptr<OlmoInferenceState> state_;
    OlmoTextGenConfig config_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string logits_output_name_;
    bool state_bound_{false};
    double last_setup_ms_{0.0};

    // Internal: generate from token IDs with sampling parameters and timing.
    struct TimedGenResult {
        std::vector<int32_t> token_ids;
        double prefill_ms{0.0};
        double decode_ms{0.0};
    };
    TimedGenResult generate_from_ids(const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
                                     const OlmoSamplingParams& params,
                                     const TextGenerationConfig& cfg);
    std::string resolve_generation_mode(const TextGenerationConfig& cfg) const;
    void reset_generation_context();

    // Run one decoder step: token_id → logits (D2H to host). Updates cache.
    void run_step(int32_t token_id, std::vector<float>& logits);

    // Decode loop (extracted for CCN).
    int32_t run_decode_loop(OlmoISampler* sampler, const OlmoSamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens, const TextGenerationConfig& cfg,
                            int32_t prompt_token_count);
    ITrtModule& bind_decoder_for_step();
    void run_prefill(const std::vector<int32_t>& input_ids, std::vector<float>& logits);
    // Execute the required family-owned batched prefill engine.
    void run_prefill_batched(const std::vector<int32_t>& input_ids, std::vector<float>& logits);
    void prime_decoder_after_batched_prefill(const std::vector<int32_t>& input_ids);
    bool should_stop_on_answer(const std::vector<int32_t>& output, int32_t prompt_token_count,
                               const TextGenerationConfig& cfg, int32_t steps,
                               int32_t stop_interval, bool is_eos) const;
};

} // namespace trtmc
