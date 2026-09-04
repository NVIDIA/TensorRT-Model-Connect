/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/k2_horizon/runtime/pipeline.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

std::string normalized_generation_mode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    std::replace(mode.begin(), mode.end(), '-', '_');
    return mode.empty() ? "auto" : mode;
}

bool has_encoder_controls(const TextGenerationConfig& cfg) {
    return cfg.source_language_token_id >= 0 || cfg.forced_bos_token_id >= 0;
}

bool has_diffusion_controls(const TextGenerationConfig& cfg) {
    return cfg.guidance_scale >= 0.0F || cfg.cfg_scale >= 0.0F || cfg.num_steps >= 0 ||
           cfg.sde_gamma >= 0.0F || cfg.block_length != 0 || cfg.confidence_threshold >= 0.0F;
}

bool has_conditioning_inputs(const TextGenerationConfig& cfg) {
    return !cfg.initial_latents.empty() || !cfg.condition_latents.empty() ||
           !cfg.condition_mask.empty() || !cfg.sampling_steps.empty() || !cfg.sde_noises.empty();
}

bool has_non_text_inputs(const TextGenerationConfig& cfg) {
    return has_encoder_controls(cfg) || has_diffusion_controls(cfg) || has_conditioning_inputs(cfg);
}

} // namespace

void k2_horizon_validate_generate_config(const TextGenerationConfig& cfg) {
    if (cfg.max_new_tokens < 0)
        throw std::invalid_argument("K2-Horizon max_new_tokens must be non-negative");
    const std::string mode = normalized_generation_mode(cfg.text_generation_mode);
    if (mode != "auto" && mode != "ar" && mode != "autoregressive") {
        throw std::invalid_argument("K2-Horizon currently supports autoregressive generation only");
    }
    if (cfg.use_chat_template) {
        throw std::invalid_argument(
            "K2-Horizon currently supports plain completion only; chat templates are disabled");
    }
    if (cfg.stop_on_boxed_answer) {
        throw std::invalid_argument(
            "K2-Horizon does not support reasoning-specific answer-stop parsing");
    }
    if (!cfg.lora_adapter_id.empty())
        throw std::invalid_argument("K2-Horizon does not support LoRA adapters");
    if (has_non_text_inputs(cfg)) {
        throw std::invalid_argument(
            "K2-Horizon received generation controls outside its plain-completion contract");
    }
}

void k2_horizon_validate_generation_inputs(const std::vector<int32_t>& token_ids,
                                           int32_t max_new_tokens, int32_t vocab_size) {
    if (vocab_size <= 0)
        throw std::invalid_argument("K2-Horizon vocabulary size must be positive");
    if (max_new_tokens > 0 && token_ids.empty())
        throw std::invalid_argument("K2-Horizon generation requires a nonempty prompt");
    for (int32_t token_id : token_ids) {
        if (token_id < 0 || token_id >= vocab_size)
            throw std::invalid_argument("K2-Horizon token ID is outside the model vocabulary");
    }
}

K2HorizonTextGenerationPipeline::K2HorizonTextGenerationPipeline(
    std::unique_ptr<ITrtModule> decoder, std::unique_ptr<K2HorizonKvCache> cache,
    K2HorizonTextGenConfig config, std::shared_ptr<ITokenizer> tokenizer)
    : decoder_(std::move(decoder)), cache_(std::move(cache)), config_(std::move(config)),
      tokenizer_(std::move(tokenizer)) {
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("K2HorizonTextGenerationPipeline: invalid decoder module");
    if (!cache_ || !cache_->ok())
        throw std::runtime_error("K2HorizonTextGenerationPipeline: invalid native KV cache");
    if (config_.vocab_size <= 0)
        throw std::invalid_argument("K2HorizonTextGenerationPipeline: invalid vocabulary size");
    if (config_.enable_cuda_graph)
        decoder_->enable_cuda_graph();
}

TextResult K2HorizonTextGenerationPipeline::generate(const std::string& prompt,
                                                     const TextGenerationConfig& cfg) {
    k2_horizon_validate_generate_config(cfg);
    if (!tokenizer_)
        throw std::runtime_error("K2HorizonTextGenerationPipeline: no tokenizer configured");

    const auto input_ids = tokenizer_->encode(prompt);
    const int32_t max_new_tokens = cfg.max_new_tokens;
    const auto params = k2_horizon_sampling_params_from_config(cfg, config_.eos_token_ids);

    last_setup_ms_ = 0.0;
    auto timed = generate_from_ids(input_ids, max_new_tokens, params, cfg);
    std::vector<int32_t> new_tokens(timed.token_ids.begin() +
                                        static_cast<std::ptrdiff_t>(input_ids.size()),
                                    timed.token_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    auto result =
        TextResult{std::move(text), std::move(new_tokens), timed.prefill_ms, timed.decode_ms};
    result.setup_ms = last_setup_ms_;
    return result;
}

K2HorizonTextGenerationPipeline::GenerationResult
K2HorizonTextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                              const TextGenerationConfig& cfg) {
    k2_horizon_validate_generate_config(cfg);
    const auto params = k2_horizon_sampling_params_from_config(cfg, config_.eos_token_ids);
    return GenerationResult{
        generate_from_ids(input_ids, cfg.max_new_tokens, params, cfg).token_ids};
}

K2HorizonTextGenerationPipeline::TimedGenResult K2HorizonTextGenerationPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const K2HorizonSamplingParams& params, const TextGenerationConfig& cfg) {
    k2_horizon_validate_generate_config(cfg);
    k2_horizon_validate_generation_inputs(input_ids, max_new_tokens, config_.vocab_size);
    k2_horizon_validate_sampling_params(params, config_.vocab_size);

    auto active_sampler = create_k2_horizon_sampler(params);
    active_sampler->reset();

    if (max_new_tokens == 0)
        return TimedGenResult{input_ids, 0.0, 0.0};

    const auto capacity = static_cast<std::size_t>(cache_->max_length());
    if (input_ids.size() > capacity ||
        static_cast<std::size_t>(max_new_tokens) > capacity - input_ids.size()) {
        throw std::invalid_argument(
            "K2-Horizon prompt and generation exceed the fixed KV cache capacity");
    }

    reset_generation_context();
    std::vector<float> logits;
    const auto prefill_start = std::chrono::steady_clock::now();
    for (int32_t token_id : input_ids)
        run_step(token_id, logits);
    const auto prefill_end = std::chrono::steady_clock::now();

    std::vector<int32_t> output = input_ids;
    const auto decode_start = std::chrono::steady_clock::now();
    run_decode_loop(*active_sampler, params, output, logits, max_new_tokens);
    const auto decode_end = std::chrono::steady_clock::now();

    return TimedGenResult{
        std::move(output),
        std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count(),
        std::chrono::duration<double, std::milli>(decode_end - decode_start).count()};
}

void K2HorizonTextGenerationPipeline::reset_generation_context() {
    const auto start = std::chrono::steady_clock::now();
    cache_->reset();
    decoder_->reset_execution_context();
    cache_->bind_to(*decoder_);
    last_setup_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void K2HorizonTextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits) {
    TensorMap inputs;
    inputs[config_.token_id_name] = Tensor{&token_id, {1}, DType::kInt32};
    cache_->prepare_step(inputs);

    const TensorMap outputs = decoder_->forward(inputs);
    const auto found = outputs.find(config_.logits_output_name);
    if (found == outputs.end()) {
        throw std::runtime_error("K2HorizonTextGenerationPipeline: logits output is missing");
    }
    const auto count = static_cast<std::size_t>(found->second.numel());
    if (count != static_cast<std::size_t>(config_.vocab_size)) {
        throw std::runtime_error(
            "K2HorizonTextGenerationPipeline: logits shape changed at runtime");
    }
    logits.resize(count);
    std::memcpy(logits.data(), found->second.data, count * sizeof(float));
    cache_->advance();
}

int32_t K2HorizonTextGenerationPipeline::run_decode_loop(K2HorizonISampler& sampler,
                                                         const K2HorizonSamplingParams& params,
                                                         std::vector<int32_t>& output,
                                                         std::vector<float>& logits,
                                                         int32_t max_new_tokens) {
    const auto start = std::chrono::steady_clock::now();
    int32_t steps = 0;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const auto sampled = sampler.sample(logits.data(), config_.vocab_size, params);
        if (sampled.token_id < 0 || sampled.token_id >= config_.vocab_size)
            throw std::runtime_error("K2-Horizon sampler returned an out-of-range token ID");
        output.push_back(sampled.token_id);
        ++steps;
        if (sampled.is_eos || k2_horizon_is_eos_token(params, sampled.token_id))
            break;
        if (step + 1 < max_new_tokens)
            run_step(sampled.token_id, logits);
    }
    const auto elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    log_decode_summary(steps, elapsed);
    return steps;
}

void K2HorizonTextGenerationPipeline::log_decode_summary(int32_t steps, double milliseconds) const {
    if (!config_.log_runtime_stats || steps <= 0)
        return;
    const double tokens_per_second = milliseconds > 0.0 ? steps * 1000.0 / milliseconds : 0.0;
    std::cerr << "[trtmc] K2-Horizon decode: " << steps << " tokens, " << milliseconds << " ms, "
              << tokens_per_second << " tok/s"
              << (decoder_->cuda_graph_active() ? " [CUDA Graph ON]" : "") << '\n';
}

} // namespace trtmc
