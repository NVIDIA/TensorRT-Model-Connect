/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct TextGenerationConfig;

enum class K2HorizonUnoGenerationMode {
    kAutoregressive,
    kLinearPsiSpec,
};

struct K2HorizonUnoResolvedGenerateConfig {
    K2HorizonUnoGenerationMode mode{K2HorizonUnoGenerationMode::kLinearPsiSpec};
    int32_t block_length{8};
};

struct K2HorizonUnoCommitDecision {
    // Tokens to publish after the currently uncached seed token.
    std::vector<int32_t> tokens;
    // Number of future-position draft tokens accepted before a rejection.
    int32_t accepted_draft_tokens{0};
    bool all_draft_tokens_accepted{false};
};

// Validate the deliberately narrow public request surface. The initial runtime
// supports batch-one greedy AR and greedy linear Psi-Spec only.
K2HorizonUnoResolvedGenerateConfig
k2_horizon_uno_resolve_generate_config(const TextGenerationConfig& config,
                                       int32_t default_block_length = 8,
                                       int32_t maximum_block_length = 8);

// Deterministic lowest-index argmax over row-major [rows, vocab_size] logits.
std::vector<int32_t> k2_horizon_uno_argmax_rows(const float* logits, int32_t rows,
                                                int32_t vocab_size);

// Exact B=1 implementation of ifm-ai/uno's deterministic_uniform noise. The
// sequence id/noise salt is fixed to zero. total_sequence_length must equal the
// prompt length plus completion_token_count.
std::vector<int32_t> k2_horizon_uno_deterministic_uniform_noise(
    const std::vector<int32_t>& prompt_token_ids, int32_t completion_token_count,
    int32_t seed_token, int32_t total_sequence_length, int32_t count, int32_t vocab_size);

// Conditional LoRA is disabled for the causal seed and enabled only for noise
// rows: [0, 1, ..., 1].
std::vector<float> k2_horizon_uno_draft_lora_mask(int32_t block_length);

// Greedy Psi-Spec verification. proposal_tokens is [clean, spec_1, ...].
// verifier_next_tokens contains the base-AR next token predicted after each
// corresponding proposal row. A rejection commits the verified correction;
// full acceptance commits the final verifier lookahead.
K2HorizonUnoCommitDecision
k2_horizon_uno_decide_greedy_commit(const std::vector<int32_t>& proposal_tokens,
                                    const std::vector<int32_t>& verifier_next_tokens);

bool k2_horizon_uno_is_eos(const std::vector<int32_t>& eos_token_ids, int32_t token_id);
void k2_horizon_uno_validate_eos_token_ids(const std::vector<int32_t>& eos_token_ids,
                                           int32_t vocab_size);

// Limit a commit to the caller's remaining budget and stop after the first EOS,
// retaining EOS itself.
void k2_horizon_uno_truncate_commit(std::vector<int32_t>& tokens, std::size_t maximum_tokens,
                                    const std::vector<int32_t>& eos_token_ids);

// With an uncached seed before the cycle, committing N new tokens leaves that
// seed plus the first N-1 committed tokens in KV. Therefore the new logical KV
// frontier is cycle_start + N.
int32_t k2_horizon_uno_retained_kv_position(int32_t cycle_start, std::size_t committed_tokens);

std::string k2_horizon_uno_format_decode_receipt(K2HorizonUnoGenerationMode mode,
                                                 int32_t block_length,
                                                 const std::string& noise_mode, int32_t forwards,
                                                 int32_t committed_tokens, int32_t lookaheads);

} // namespace trtmc
