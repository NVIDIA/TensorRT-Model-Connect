/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Model-owned decoder text pipeline.
//
// Composes: ITrtModule (decoder) + NemotronLabsDiffusionKvCache + ITokenizer for this runtime
// plugin. Architecture-specific behavior remains in this model directory and
// in the TRT engine emitted by the matching family builder.

#include "families/nemotron_labs_diffusion/runtime/inference_state.h"
#include "families/nemotron_labs_diffusion/runtime/sampler.h"
#include "families/nemotron_labs_diffusion/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class NemotronLabsDiffusionKvCache;

struct NemotronLabsDiffusionTextGenConfig {
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
    int32_t mask_token_id{-1};
    int32_t diffusion_block_length{32};
    bool supports_text_diffusion{false};
};

class NemotronLabsDiffusionTextGenerationPipeline final : public ITextGeneration {
  public:
    NemotronLabsDiffusionTextGenerationPipeline(
        std::unique_ptr<ITrtModule> decoder,
        std::unique_ptr<NemotronLabsDiffusionInferenceState> state,
        NemotronLabsDiffusionTextGenConfig config, std::shared_ptr<ITokenizer> tokenizer,
        std::unique_ptr<ITrtModule> prefill,
        std::unique_ptr<ITrtModule> linear_spec_lora_prefill = nullptr,
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
    std::unique_ptr<ITrtModule> linear_spec_lora_prefill_;
    std::unique_ptr<NemotronLabsDiffusionInferenceState> state_;
    NemotronLabsDiffusionTextGenConfig config_;
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
                                     const NemotronLabsDiffusionSamplingParams& params,
                                     const TextGenerationConfig& cfg);
    TimedGenResult generate_diffusion_from_ids(const std::vector<int32_t>& input_ids,
                                               int32_t max_new_tokens,
                                               const NemotronLabsDiffusionSamplingParams& params,
                                               const TextGenerationConfig& cfg);
    TimedGenResult generate_linear_spec_from_ids(const std::vector<int32_t>& input_ids,
                                                 int32_t max_new_tokens,
                                                 const NemotronLabsDiffusionSamplingParams& params,
                                                 const TextGenerationConfig& cfg,
                                                 bool use_lora_draft);
    std::string resolve_generation_mode(const TextGenerationConfig& cfg) const;
    void reset_generation_context();
    ITrtModule& require_block_prefill(int32_t sq, ITrtModule* prefill_override);
    NemotronLabsDiffusionKvCache& require_block_kv_cache();
    void copy_block_logits(const TensorMap& outputs, std::vector<float>& logits) const;
    void append_prefill_kv(NemotronLabsDiffusionKvCache& kv, ITrtModule& prefill, int32_t sq);
    int32_t resolve_text_diffusion_block_length(const TextGenerationConfig& cfg,
                                                int32_t max_new_tokens,
                                                bool require_divisible) const;
    int32_t seed_next_token_from_prefill(const std::vector<int32_t>& input_ids,
                                         std::vector<float>& logits, int32_t vocab);
    void fill_diffusion_block(std::vector<int32_t>& block, std::vector<float>& logits,
                              int32_t block_len, int32_t vocab, bool use_threshold,
                              float threshold);
    int32_t verify_diffusion_block(const std::vector<int32_t>& block, std::vector<float>& logits,
                                   int32_t block_len, int32_t vocab);
    bool append_tokens_until_eos(const std::vector<int32_t>& tokens, std::vector<int32_t>& output,
                                 const NemotronLabsDiffusionSamplingParams& params) const;
    void fill_linear_spec_block(std::vector<int32_t>& block, std::vector<float>& logits,
                                int32_t block_len, int32_t vocab, bool threshold_enabled,
                                float threshold, bool use_lora_draft);
    std::vector<int32_t> verify_linear_spec_block(const std::vector<int32_t>& block,
                                                  std::vector<float>& logits, int32_t block_len,
                                                  int32_t vocab);
    static int32_t count_linear_spec_accepts(const std::vector<int32_t>& ar_tokens,
                                             const std::vector<int32_t>& block);
    bool append_linear_spec_tokens(const std::vector<int32_t>& ar_tokens, int32_t emit_count,
                                   std::vector<int32_t>& output, int32_t& generated,
                                   const NemotronLabsDiffusionSamplingParams& params) const;

    // Run one decoder step: token_id → logits (D2H to host). Updates cache.
    void run_step(int32_t token_id, std::vector<float>& logits);

    // Decode loop (extracted for CCN).
    int32_t run_decode_loop(NemotronLabsDiffusionISampler* sampler,
                            const NemotronLabsDiffusionSamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens, const TextGenerationConfig& cfg,
                            int32_t prompt_token_count);
    ITrtModule& bind_decoder_for_step();
    void run_prefill(const std::vector<int32_t>& input_ids, std::vector<float>& logits);
    void run_prefill_block(const std::vector<int32_t>& input_ids, bool bidirectional,
                           bool append_kv, std::vector<float>& logits,
                           ITrtModule* prefill_override = nullptr);
    // Execute the required family-owned batched prefill engine.
    void run_prefill_batched(const std::vector<int32_t>& input_ids, std::vector<float>& logits);
    void prime_decoder_after_batched_prefill(const std::vector<int32_t>& input_ids);
    bool should_stop_on_answer(const std::vector<int32_t>& output, int32_t prompt_token_count,
                               const TextGenerationConfig& cfg, int32_t steps,
                               int32_t stop_interval, bool is_eos) const;
};

} // namespace trtmc
