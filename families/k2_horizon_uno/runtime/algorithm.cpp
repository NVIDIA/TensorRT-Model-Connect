/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/k2_horizon_uno/runtime/algorithm.h"

#include "trtmc/task.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace trtmc {
namespace {

constexpr std::uint64_t kPromptSeed = UINT64_C(0xD6E8FEB86659FD93);
constexpr std::uint64_t kPromptMultiplier = UINT64_C(0x9E3779B185EBCA87);
constexpr std::uint64_t kNoiseSaltMultiplier = UINT64_C(0xD1B54A32D192ED03);
constexpr std::uint64_t kCompletionMultiplier = UINT64_C(0xC2B2AE3D27D4EB4F);
constexpr std::uint64_t kSeedTokenMultiplier = UINT64_C(0x165667B19E3779F9);
constexpr std::uint64_t kSequenceLengthMultiplier = UINT64_C(0x85EBCA77C2B2AE63);
constexpr std::uint64_t kSlotMultiplier = UINT64_C(0x27D4EB2F165667C5);

std::uint64_t mix_u64(std::uint64_t value) {
    value = (value ^ (value >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31U);
}

bool supported_noise_mode(const std::string& mode) {
    return mode == "deterministic_uniform";
}

std::string normalize_mode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    std::replace(mode.begin(), mode.end(), '-', '_');
    return mode.empty() ? "auto" : mode;
}

bool uses_greedy_sampling(const TextGenerationConfig& config) {
    return config.temperature == 0.0F || config.top_p == 0.0F ||
           (config.top_k == 1 && config.top_p == 1.0F);
}

bool has_diffusion_scalar_controls(const TextGenerationConfig& config) {
    return config.guidance_scale >= 0.0F || config.cfg_scale >= 0.0F || config.num_steps >= 0 ||
           config.sde_gamma >= 0.0F || config.confidence_threshold >= 0.0F;
}

bool has_conditioning_inputs(const TextGenerationConfig& config) {
    return !config.initial_latents.empty() || !config.condition_latents.empty() ||
           !config.condition_mask.empty() || !config.sampling_steps.empty() ||
           !config.sde_noises.empty();
}

bool has_diffusion_controls(const TextGenerationConfig& config) {
    return has_diffusion_scalar_controls(config) || has_conditioning_inputs(config);
}

void validate_decoder_request_shape(const TextGenerationConfig& config) {
    if (config.max_new_tokens < 0)
        throw std::invalid_argument("K2-Horizon-Uno max_new_tokens must be non-negative");
    if (config.source_language_token_id >= 0)
        throw std::invalid_argument("K2-Horizon-Uno supports one decoder-only sample");
    if (config.forced_bos_token_id >= 0)
        throw std::invalid_argument("K2-Horizon-Uno supports one decoder-only sample");
}

void validate_finite_sampling_controls(const TextGenerationConfig& config) {
    if (!std::isfinite(config.temperature))
        throw std::invalid_argument("K2-Horizon-Uno sampling controls are outside their range");
    if (!std::isfinite(config.top_p))
        throw std::invalid_argument("K2-Horizon-Uno sampling controls are outside their range");
    if (!std::isfinite(config.min_p))
        throw std::invalid_argument("K2-Horizon-Uno sampling controls are outside their range");
    if (!std::isfinite(config.repetition_penalty))
        throw std::invalid_argument("K2-Horizon-Uno sampling controls are outside their range");
}

void validate_sampling_ranges(const TextGenerationConfig& config) {
    if (config.temperature < 0.0F)
        throw std::invalid_argument("K2-Horizon-Uno sampling controls are outside their range");
    if (config.top_p < 0.0F)
        throw std::invalid_argument("K2-Horizon-Uno sampling controls are outside their range");
    if (config.top_p > 1.0F)
        throw std::invalid_argument("K2-Horizon-Uno sampling controls are outside their range");
    if (config.min_p < 0.0F)
        throw std::invalid_argument("K2-Horizon-Uno sampling controls are outside their range");
    if (config.min_p > 1.0F)
        throw std::invalid_argument("K2-Horizon-Uno sampling controls are outside their range");
    if (config.repetition_penalty <= 0.0F)
        throw std::invalid_argument("K2-Horizon-Uno sampling controls are outside their range");
}

void validate_greedy_sampling_contract(const TextGenerationConfig& config) {
    if (!uses_greedy_sampling(config)) {
        throw std::invalid_argument(
            "K2-Horizon-Uno currently supports deterministic greedy generation only");
    }
    if (config.min_p != 0.0F)
        throw std::invalid_argument("K2-Horizon-Uno does not support min-p or repetition penalty");
    if (config.repetition_penalty != 1.0F)
        throw std::invalid_argument("K2-Horizon-Uno does not support min-p or repetition penalty");
    if (config.seed != -1) {
        throw std::invalid_argument(
            "K2-Horizon-Uno deterministic noise does not accept a seed override");
    }
}

void validate_optional_request_controls(const TextGenerationConfig& config) {
    if (config.use_chat_template && !config.enable_thinking) {
        throw std::invalid_argument(
            "K2-Horizon-Uno chat supports only publisher high-reasoning mode");
    }
    if (config.stop_on_boxed_answer)
        throw std::invalid_argument("K2-Horizon-Uno does not support answer-stop parsing");
    if (!config.lora_adapter_id.empty()) {
        throw std::invalid_argument(
            "K2-Horizon-Uno uses its pinned bundled adapter and rejects external LoRA IDs");
    }
    if (has_diffusion_controls(config)) {
        throw std::invalid_argument(
            "K2-Horizon-Uno received controls outside its linear Psi-Spec contract");
    }
}

void validate_common_controls(const TextGenerationConfig& config) {
    validate_decoder_request_shape(config);
    validate_finite_sampling_controls(config);
    validate_sampling_ranges(config);
    validate_greedy_sampling_contract(config);
    validate_optional_request_controls(config);
}

void validate_block_bounds(int32_t default_block_length, int32_t maximum_block_length) {
    if (default_block_length <= 0)
        throw std::invalid_argument("K2-Horizon-Uno bundle block-length bounds are invalid");
    if (maximum_block_length <= 0)
        throw std::invalid_argument("K2-Horizon-Uno bundle block-length bounds are invalid");
    if (default_block_length > maximum_block_length)
        throw std::invalid_argument("K2-Horizon-Uno bundle block-length bounds are invalid");
}

bool is_uno_mode(const std::string& mode) {
    static const std::vector<std::string> modes{"auto", "uno", "linear_psi_spec",
                                                "linear_spec_lora"};
    return std::find(modes.begin(), modes.end(), mode) != modes.end();
}

bool is_ar_mode(const std::string& mode) {
    static const std::vector<std::string> modes{"ar", "autoregressive"};
    return std::find(modes.begin(), modes.end(), mode) != modes.end();
}

int32_t resolve_uno_block_length(const TextGenerationConfig& config, int32_t default_block_length,
                                 int32_t maximum_block_length) {
    const int32_t block_length =
        config.block_length == 0 ? default_block_length : config.block_length;
    if (block_length <= 0 || block_length > maximum_block_length) {
        throw std::invalid_argument("K2-Horizon-Uno block_length must be in [1, " +
                                    std::to_string(maximum_block_length) + "]");
    }
    return block_length;
}

void validate_ar_block_length(int32_t block_length) {
    if (block_length == 0)
        return;
    if (block_length == 1)
        return;
    throw std::invalid_argument("K2-Horizon-Uno autoregressive mode requires block_length 0 or 1");
}

void validate_deterministic_noise_scalars(int32_t completion_token_count, int32_t seed_token,
                                          int32_t total_sequence_length, int32_t count,
                                          int32_t vocab_size) {
    if (vocab_size <= 1)
        throw std::invalid_argument("K2-Horizon-Uno noise range requires vocab_size > 1");
    if (completion_token_count < 0)
        throw std::invalid_argument(
            "K2-Horizon-Uno deterministic noise counts must be non-negative");
    if (total_sequence_length < 0)
        throw std::invalid_argument(
            "K2-Horizon-Uno deterministic noise counts must be non-negative");
    if (count < 0)
        throw std::invalid_argument(
            "K2-Horizon-Uno deterministic noise counts must be non-negative");
    if (seed_token < 0 || seed_token >= vocab_size)
        throw std::invalid_argument("K2-Horizon-Uno deterministic noise seed is out of range");
}

void validate_sequence_accounting(const std::vector<int32_t>& prompt_token_ids,
                                  int32_t completion_token_count, int32_t total_sequence_length) {
    if (prompt_token_ids.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument(
            "K2-Horizon-Uno deterministic noise sequence length is inconsistent");
    const auto expected =
        static_cast<std::int64_t>(prompt_token_ids.size()) + completion_token_count;
    if (expected > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument(
            "K2-Horizon-Uno deterministic noise sequence length is inconsistent");
    if (total_sequence_length != expected)
        throw std::invalid_argument(
            "K2-Horizon-Uno deterministic noise sequence length is inconsistent");
}

std::uint64_t prompt_noise_seed(const std::vector<int32_t>& prompt_token_ids, int32_t vocab_size) {
    std::uint64_t seed = kPromptSeed;
    for (int32_t token : prompt_token_ids) {
        if (token < 0 || token >= vocab_size)
            throw std::invalid_argument(
                "K2-Horizon-Uno deterministic noise prompt token is out of range");
        seed = mix_u64(seed ^ static_cast<std::uint64_t>(token));
    }
    return seed;
}

std::uint64_t deterministic_noise_base(std::uint64_t sequence_seed, int32_t completion_token_count,
                                       int32_t seed_token, int32_t total_sequence_length) {
    // B=1 fixes noise_salt/sequence id to zero.
    return sequence_seed * kPromptMultiplier + UINT64_C(0) * kNoiseSaltMultiplier +
           static_cast<std::uint64_t>(completion_token_count) * kCompletionMultiplier +
           static_cast<std::uint64_t>(seed_token) * kSeedTokenMultiplier +
           static_cast<std::uint64_t>(total_sequence_length) * kSequenceLengthMultiplier;
}

} // namespace

K2HorizonUnoResolvedGenerateConfig
k2_horizon_uno_resolve_generate_config(const TextGenerationConfig& config,
                                       int32_t default_block_length, int32_t maximum_block_length) {
    validate_common_controls(config);
    validate_block_bounds(default_block_length, maximum_block_length);

    const std::string mode = normalize_mode(config.text_generation_mode);
    K2HorizonUnoResolvedGenerateConfig resolved;
    if (is_uno_mode(mode)) {
        resolved.mode = K2HorizonUnoGenerationMode::kLinearPsiSpec;
        resolved.block_length =
            resolve_uno_block_length(config, default_block_length, maximum_block_length);
        return resolved;
    }
    if (is_ar_mode(mode)) {
        validate_ar_block_length(config.block_length);
        resolved.mode = K2HorizonUnoGenerationMode::kAutoregressive;
        resolved.block_length = 1;
        return resolved;
    }
    throw std::invalid_argument("Unsupported K2-Horizon-Uno generation mode: " + mode);
}

std::vector<int32_t> k2_horizon_uno_argmax_rows(const float* logits, int32_t rows,
                                                int32_t vocab_size) {
    if (logits == nullptr || rows <= 0 || vocab_size <= 1)
        throw std::invalid_argument("K2-Horizon-Uno argmax requires nonempty row logits");
    if (static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(vocab_size) >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("K2-Horizon-Uno logits element count overflows size_t");
    }

    std::vector<int32_t> result;
    result.reserve(static_cast<std::size_t>(rows));
    for (int32_t row = 0; row < rows; ++row) {
        const float* row_logits =
            logits + static_cast<std::size_t>(row) * static_cast<std::size_t>(vocab_size);
        if (!std::isfinite(row_logits[0]))
            throw std::runtime_error("K2-Horizon-Uno received non-finite logits");
        int32_t best = 0;
        for (int32_t token = 1; token < vocab_size; ++token) {
            if (!std::isfinite(row_logits[token]))
                throw std::runtime_error("K2-Horizon-Uno received non-finite logits");
            if (row_logits[token] > row_logits[best])
                best = token;
        }
        result.push_back(best);
    }
    return result;
}

std::vector<int32_t> k2_horizon_uno_deterministic_uniform_noise(
    const std::vector<int32_t>& prompt_token_ids, int32_t completion_token_count,
    int32_t seed_token, int32_t total_sequence_length, int32_t count, int32_t vocab_size) {
    validate_deterministic_noise_scalars(completion_token_count, seed_token, total_sequence_length,
                                         count, vocab_size);
    validate_sequence_accounting(prompt_token_ids, completion_token_count, total_sequence_length);
    const std::uint64_t sequence_seed = prompt_noise_seed(prompt_token_ids, vocab_size);

    // Unsigned arithmetic intentionally wraps modulo 2^64, matching Python's
    // explicit masking in ifm-ai/uno.
    const std::uint64_t base = deterministic_noise_base(sequence_seed, completion_token_count,
                                                        seed_token, total_sequence_length);
    const std::uint64_t span = static_cast<std::uint64_t>(vocab_size - 1);
    std::vector<int32_t> noise;
    noise.reserve(static_cast<std::size_t>(count));
    for (int32_t slot = 0; slot < count; ++slot) {
        const std::uint64_t mixed =
            mix_u64(base + static_cast<std::uint64_t>(slot) * kSlotMultiplier);
        noise.push_back(1 + static_cast<int32_t>(mixed % span));
    }
    return noise;
}

std::vector<float> k2_horizon_uno_draft_lora_mask(int32_t block_length) {
    if (block_length <= 0)
        throw std::invalid_argument("K2-Horizon-Uno LoRA mask must be nonempty");
    std::vector<float> mask(static_cast<std::size_t>(block_length), 1.0F);
    mask.front() = 0.0F;
    return mask;
}

K2HorizonUnoCommitDecision
k2_horizon_uno_decide_greedy_commit(const std::vector<int32_t>& proposal_tokens,
                                    const std::vector<int32_t>& verifier_next_tokens) {
    if (proposal_tokens.empty() || proposal_tokens.size() != verifier_next_tokens.size()) {
        throw std::invalid_argument(
            "K2-Horizon-Uno verification requires equal nonempty proposal and verifier rows");
    }

    K2HorizonUnoCommitDecision decision;
    const std::size_t future_count = proposal_tokens.size() - 1;
    for (std::size_t index = 0; index < future_count; ++index) {
        if (verifier_next_tokens[index] == proposal_tokens[index + 1]) {
            ++decision.accepted_draft_tokens;
            continue;
        }

        decision.tokens.insert(decision.tokens.end(), proposal_tokens.begin(),
                               proposal_tokens.begin() + static_cast<std::ptrdiff_t>(index + 1));
        decision.tokens.push_back(verifier_next_tokens[index]);
        return decision;
    }

    decision.all_draft_tokens_accepted = true;
    decision.tokens = proposal_tokens;
    decision.tokens.push_back(verifier_next_tokens.back());
    return decision;
}

bool k2_horizon_uno_is_eos(const std::vector<int32_t>& eos_token_ids, int32_t token_id) {
    return std::find(eos_token_ids.begin(), eos_token_ids.end(), token_id) != eos_token_ids.end();
}

void k2_horizon_uno_validate_eos_token_ids(const std::vector<int32_t>& eos_token_ids,
                                           int32_t vocab_size) {
    if (vocab_size <= 1 || eos_token_ids.empty())
        throw std::invalid_argument("K2-Horizon-Uno requires a nonempty in-range EOS set");
    std::vector<int32_t> seen;
    seen.reserve(eos_token_ids.size());
    for (int32_t token : eos_token_ids) {
        if (token < 0 || token >= vocab_size)
            throw std::invalid_argument("K2-Horizon-Uno EOS token is outside vocabulary");
        if (std::find(seen.begin(), seen.end(), token) != seen.end())
            throw std::invalid_argument("K2-Horizon-Uno EOS token set contains duplicates");
        seen.push_back(token);
    }
}

void k2_horizon_uno_truncate_commit(std::vector<int32_t>& tokens, std::size_t maximum_tokens,
                                    const std::vector<int32_t>& eos_token_ids) {
    if (tokens.size() > maximum_tokens)
        tokens.resize(maximum_tokens);
    const auto eos = std::find_if(tokens.begin(), tokens.end(), [&](int32_t token) {
        return k2_horizon_uno_is_eos(eos_token_ids, token);
    });
    if (eos != tokens.end())
        tokens.erase(eos + 1, tokens.end());
}

int32_t k2_horizon_uno_retained_kv_position(int32_t cycle_start, std::size_t committed_tokens) {
    if (cycle_start < 0 || committed_tokens == 0)
        throw std::invalid_argument(
            "K2-Horizon-Uno retained KV position requires a nonempty commit");
    if (committed_tokens >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max() - cycle_start)) {
        throw std::overflow_error("K2-Horizon-Uno retained KV position overflows int32");
    }
    return cycle_start + static_cast<int32_t>(committed_tokens);
}

std::string k2_horizon_uno_format_decode_receipt(K2HorizonUnoGenerationMode mode,
                                                 int32_t block_length,
                                                 const std::string& noise_mode, int32_t forwards,
                                                 int32_t committed_tokens, int32_t lookaheads) {
    if (block_length <= 0 || forwards < 0 || committed_tokens < 0 || lookaheads < 0)
        throw std::invalid_argument("K2-Horizon-Uno decode receipt counters are invalid");
    if (!supported_noise_mode(noise_mode))
        throw std::invalid_argument("K2-Horizon-Uno decode receipt noise mode is invalid");
    std::ostringstream receipt;
    receipt << "[trtmc.k2_horizon_uno.decode] mode="
            << (mode == K2HorizonUnoGenerationMode::kLinearPsiSpec ? "linear_spec_lora" : "ar")
            << " block_length=" << block_length << " noise_mode=" << noise_mode
            << " forwards=" << forwards << " committed_tokens=" << committed_tokens
            << " lookaheads=" << lookaheads;
    return receipt.str();
}

} // namespace trtmc
