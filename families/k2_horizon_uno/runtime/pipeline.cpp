/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/k2_horizon_uno/runtime/pipeline.h"

#include "families/k2_horizon_uno/runtime/chat_template.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

constexpr int32_t kMaximumBlockLength = 8;
constexpr int32_t kVocabSize = 250624;

int32_t checked_size_to_int32(std::size_t value, const char* label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::overflow_error(std::string("K2-Horizon-Uno ") + label + " exceeds int32");
    return static_cast<int32_t>(value);
}

void validate_pipeline_components(const std::unique_ptr<ITrtModule>& decoder,
                                  const std::unique_ptr<K2HorizonUnoKvCache>& cache) {
    if (!decoder || !decoder->ok())
        throw std::runtime_error("K2HorizonUnoTextGenerationPipeline: invalid decoder module");
    if (!cache || !cache->ok())
        throw std::runtime_error("K2HorizonUnoTextGenerationPipeline: invalid native KV cache");
}

void validate_generation_inputs(const std::vector<int32_t>& token_ids, int32_t max_new_tokens) {
    if (max_new_tokens > 0 && token_ids.empty())
        throw std::invalid_argument("K2-Horizon-Uno generation requires a nonempty prompt");
    for (int32_t token : token_ids) {
        if (token < 0 || token >= kVocabSize)
            throw std::invalid_argument("K2-Horizon-Uno token is outside the vocabulary");
    }
}

} // namespace

K2HorizonUnoTextGenerationPipeline::K2HorizonUnoTextGenerationPipeline(
    std::unique_ptr<ITrtModule> decoder, std::unique_ptr<K2HorizonUnoKvCache> cache,
    std::shared_ptr<ITokenizer> tokenizer)
    : decoder_(std::move(decoder)), cache_(std::move(cache)), tokenizer_(std::move(tokenizer)) {
    validate_pipeline_components(decoder_, cache_);
}

TextResult K2HorizonUnoTextGenerationPipeline::generate(const std::string& prompt,
                                                        const TextGenerationConfig& cfg) {
    (void)k2_horizon_uno_resolve_generate_config(cfg);
    if (!tokenizer_)
        throw std::runtime_error("K2HorizonUnoTextGenerationPipeline: no tokenizer configured");
    if (cfg.use_chat_template && cfg.eos_token_id >= 0) {
        throw std::invalid_argument(
            "K2-Horizon-Uno chat requires the publisher EOS set from the bundle");
    }

    const std::string effective_prompt =
        cfg.use_chat_template ? k2_horizon_uno_apply_chat_template(
                                    kK2HorizonUnoPublisherChatTemplateFormat, prompt, "high")
                              : prompt;
    const auto input_ids = tokenizer_->encode(effective_prompt);

    last_setup_ms_ = 0.0;
    auto generated = generate_from_ids(input_ids, cfg);
    std::vector<int32_t> new_tokens(generated.token_ids.begin() +
                                        static_cast<std::ptrdiff_t>(input_ids.size()),
                                    generated.token_ids.end());
    TextResult result{tokenizer_->decode(new_tokens), std::move(new_tokens), generated.prefill_ms,
                      generated.decode_ms};
    result.setup_ms = last_setup_ms_;
    log_decode_receipt(generated.stats);
    return result;
}

K2HorizonUnoTextGenerationPipeline::TimedGenResult
K2HorizonUnoTextGenerationPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                                      const TextGenerationConfig& cfg) {
    const auto resolved = k2_horizon_uno_resolve_generate_config(cfg);
    validate_generation_inputs(input_ids, cfg.max_new_tokens);
    // Validate request overrides before resetting state or executing the engine.
    (void)effective_eos_ids(cfg);
    // The current seed is not cached, so linear verification reaches at most
    // prompt_tokens + max_new_tokens, including its transient lookahead row.
    if (input_ids.size() > static_cast<std::size_t>(cache_->max_length()) ||
        static_cast<std::size_t>(cfg.max_new_tokens) >
            static_cast<std::size_t>(cache_->max_length()) - input_ids.size()) {
        throw std::invalid_argument(
            "K2-Horizon-Uno prompt and generation exceed fixed KV capacity");
    }

    if (cfg.max_new_tokens == 0) {
        TimedGenResult result;
        result.token_ids = input_ids;
        result.stats.mode = resolved.mode;
        result.stats.block_length = resolved.block_length;
        return result;
    }
    if (resolved.mode == K2HorizonUnoGenerationMode::kAutoregressive)
        return generate_ar(input_ids, cfg, resolved);
    return generate_linear_psi_spec(input_ids, cfg, resolved);
}

K2HorizonUnoTextGenerationPipeline::TimedGenResult K2HorizonUnoTextGenerationPipeline::generate_ar(
    const std::vector<int32_t>& input_ids, const TextGenerationConfig& cfg,
    const K2HorizonUnoResolvedGenerateConfig& resolved) {
    using Clock = std::chrono::steady_clock;
    reset_generation_context();
    std::vector<float> logits;
    const auto prefill_start = Clock::now();
    run_prefill(input_ids, logits);
    const auto prefill_end = Clock::now();

    const auto eos_ids = effective_eos_ids(cfg);
    std::vector<int32_t> output = input_ids;
    DecodeStats stats;
    stats.mode = resolved.mode;
    stats.block_length = 1;
    const auto decode_start = Clock::now();
    for (int32_t generated = 0; generated < cfg.max_new_tokens; ++generated) {
        const int32_t token = k2_horizon_uno_argmax_rows(logits.data(), 1, kVocabSize).front();
        output.push_back(token);
        ++stats.committed_tokens;
        if (k2_horizon_uno_is_eos(eos_ids, token))
            break;
        if (generated + 1 < cfg.max_new_tokens) {
            run_block({token}, {0.0F}, logits);
            ++stats.forwards;
        }
    }
    const auto decode_end = Clock::now();
    return TimedGenResult{
        std::move(output),
        std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count(),
        std::chrono::duration<double, std::milli>(decode_end - decode_start).count(), stats};
}

K2HorizonUnoTextGenerationPipeline::TimedGenResult
K2HorizonUnoTextGenerationPipeline::generate_linear_psi_spec(
    const std::vector<int32_t>& input_ids, const TextGenerationConfig& cfg,
    const K2HorizonUnoResolvedGenerateConfig& resolved) {
    using Clock = std::chrono::steady_clock;
    reset_generation_context();
    std::vector<float> logits;
    const auto prefill_start = Clock::now();
    run_prefill(input_ids, logits);
    const auto prefill_end = Clock::now();

    const auto eos_ids = effective_eos_ids(cfg);
    std::vector<int32_t> output = input_ids;
    DecodeStats stats;
    stats.mode = resolved.mode;
    stats.block_length = resolved.block_length;
    const auto decode_start = Clock::now();
    int32_t seed_token = k2_horizon_uno_argmax_rows(logits.data(), 1, kVocabSize).front();
    output.push_back(seed_token);
    ++stats.committed_tokens;
    int32_t generated_tokens = 1;

    while (generated_tokens < cfg.max_new_tokens && !k2_horizon_uno_is_eos(eos_ids, seed_token)) {
        const int32_t remaining = cfg.max_new_tokens - generated_tokens;
        const int32_t block_length = std::min(resolved.block_length, remaining);
        const int32_t cycle_start = cache_->position();

        std::vector<int32_t> draft{seed_token};
        auto noise = k2_horizon_uno_deterministic_uniform_noise(
            input_ids, generated_tokens, seed_token,
            checked_size_to_int32(output.size(), "sequence length"), block_length - 1, kVocabSize);
        draft.insert(draft.end(), noise.begin(), noise.end());
        run_block(draft, k2_horizon_uno_draft_lora_mask(block_length), logits);
        ++stats.forwards;
        auto proposals = k2_horizon_uno_argmax_rows(logits.data(), block_length, kVocabSize);

        // Draft K/V for the seed is valid base-model state; noise suffix rows
        // are temporary and become invisible before verification.
        cache_->rollback(cycle_start + 1);
        run_block(proposals, std::vector<float>(static_cast<std::size_t>(block_length), 0.0F),
                  logits);
        ++stats.forwards;
        const auto verifier = k2_horizon_uno_argmax_rows(logits.data(), block_length, kVocabSize);

        auto decision = k2_horizon_uno_decide_greedy_commit(proposals, verifier);
        if (decision.all_draft_tokens_accepted && block_length > 1)
            ++stats.lookaheads;
        k2_horizon_uno_truncate_commit(decision.tokens, static_cast<std::size_t>(remaining),
                                       eos_ids);
        if (decision.tokens.empty())
            throw std::runtime_error("K2-Horizon-Uno produced an empty verified commit");

        cache_->rollback(k2_horizon_uno_retained_kv_position(cycle_start, decision.tokens.size()));
        generated_tokens += checked_size_to_int32(decision.tokens.size(), "commit length");
        stats.committed_tokens += checked_size_to_int32(decision.tokens.size(), "commit length");
        output.insert(output.end(), decision.tokens.begin(), decision.tokens.end());
        seed_token = decision.tokens.back();
    }

    const auto decode_end = Clock::now();
    return TimedGenResult{
        std::move(output),
        std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count(),
        std::chrono::duration<double, std::milli>(decode_end - decode_start).count(), stats};
}

void K2HorizonUnoTextGenerationPipeline::reset_generation_context() {
    const auto start = std::chrono::steady_clock::now();
    cache_->reset();
    decoder_->reset_execution_context();
    cache_->bind_to(*decoder_);
    last_setup_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void K2HorizonUnoTextGenerationPipeline::run_prefill(const std::vector<int32_t>& input_ids,
                                                     std::vector<float>& logits) {
    const std::size_t chunk = kMaximumBlockLength;
    std::size_t offset = 0;
    while (input_ids.size() - offset > 1) {
        const std::size_t rows = std::min(chunk, input_ids.size() - offset - 1);
        const auto begin = input_ids.begin() + static_cast<std::ptrdiff_t>(offset);
        const std::vector<int32_t> tokens(begin, begin + static_cast<std::ptrdiff_t>(rows));
        run_block(tokens, std::vector<float>(rows, 0.0F), logits);
        offset += rows;
    }
    // Keep the final row isolated because seed selection consumes logits row zero.
    run_block({input_ids[offset]}, {0.0F}, logits);
}

void K2HorizonUnoTextGenerationPipeline::run_block(const std::vector<int32_t>& token_ids,
                                                   const std::vector<float>& lora_mask,
                                                   std::vector<float>& logits) {
    if (token_ids.empty() || token_ids.size() != lora_mask.size() ||
        token_ids.size() > static_cast<std::size_t>(kMaximumBlockLength)) {
        throw std::invalid_argument("K2-Horizon-Uno block input contract is invalid");
    }
    const int32_t rows = checked_size_to_int32(token_ids.size(), "block length");
    TensorMap inputs;
    inputs["token_id"] = Tensor{const_cast<int32_t*>(token_ids.data()), {rows}, DType::kInt32};
    inputs["lora_mask"] = Tensor{const_cast<float*>(lora_mask.data()), {rows}, DType::kFloat32};
    cache_->prepare_block(inputs, rows);

    const TensorMap outputs = decoder_->forward(inputs);
    const auto found = outputs.find("logits");
    if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
        found->second.shape != std::vector<int64_t>{rows, kVocabSize}) {
        throw std::runtime_error(
            "K2-Horizon-Uno engine must return float32 full logits [S,vocab_size]");
    }
    const auto count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(kVocabSize);
    logits.resize(count);
    std::memcpy(logits.data(), found->second.data, count * sizeof(float));
    cache_->advance(rows);
}

std::vector<int32_t>
K2HorizonUnoTextGenerationPipeline::effective_eos_ids(const TextGenerationConfig& cfg) const {
    std::vector<int32_t> result = cfg.eos_token_id >= 0 ? std::vector<int32_t>{cfg.eos_token_id}
                                                        : std::vector<int32_t>{1, 250019};
    k2_horizon_uno_validate_eos_token_ids(result, kVocabSize);
    return result;
}

void K2HorizonUnoTextGenerationPipeline::log_decode_receipt(const DecodeStats& stats) const {
    std::cerr << k2_horizon_uno_format_decode_receipt(stats.mode, stats.block_length,
                                                      "deterministic_uniform", stats.forwards,
                                                      stats.committed_tokens, stats.lookaheads)
              << '\n';
}

} // namespace trtmc
