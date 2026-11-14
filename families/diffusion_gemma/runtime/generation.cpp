/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/diffusion_gemma/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace trtmc::diffusion_gemma {
namespace {

const internal::ConfigField kGenerationFields[] = {
    {"max_new_tokens", internal::ConfigKind::I64, std::nullopt,
     "Generation budget; diffusion emits complete canvases, as in the reference model."},
    {"seed", internal::ConfigKind::I64, std::int64_t{0}, "Native diffusion sampler random seed."},
};

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

std::string trim(std::string value) {
    const auto start = value.find_first_not_of(" \r\n\t\f\v");
    if (start == std::string::npos)
        return {};
    const auto end = value.find_last_not_of(" \r\n\t\f\v");
    return value.substr(start, end - start + 1);
}

void check_legacy_options(const TextGenerationConfig& config) {
    if (config.temperature != 1.0F || config.top_k != 1 || config.top_p != 1.0F ||
        config.min_p != 0.0F || config.repetition_penalty != 1.0F)
        throw std::invalid_argument(
            "diffusion_gemma uses the checkpoint's entropy-bound sampler and temperature schedule");
    if (config.source_language_token_id != -1 || config.forced_bos_token_id != -1 ||
        config.guidance_scale != -1.0F || config.cfg_scale != -1.0F || config.sde_gamma != -1.0F ||
        !config.initial_latents.empty() || !config.condition_latents.empty() ||
        !config.condition_mask.empty() || !config.sampling_steps.empty() ||
        !config.sde_noises.empty() || config.eos_token_id != -1 ||
        config.text_generation_mode != "auto" || config.block_length != 0 ||
        config.confidence_threshold != -1.0F || config.stop_on_boxed_answer ||
        !config.lora_adapter_id.empty())
        throw std::invalid_argument(
            "diffusion_gemma does not support the supplied legacy generation option");
    if (!config.use_chat_template && !config.system_prompt.empty())
        throw std::invalid_argument(
            "diffusion_gemma system_prompt requires chat-template formatting");
}

} // namespace

TextResult Pipeline::generate_ids(const std::vector<std::int32_t>& prompt,
                                  std::int32_t max_new_tokens, std::int64_t seed,
                                  std::int32_t max_steps) {
    validate_ids(prompt);
    if (max_new_tokens < 1 || max_steps < 1)
        throw std::invalid_argument(
            "diffusion_gemma generation and denoising budgets must be positive");
    const auto canvas_count =
        (static_cast<std::int64_t>(max_new_tokens) + config_.canvas_length - 1) /
        config_.canvas_length;
    if (static_cast<std::int64_t>(prompt.size()) + canvas_count * config_.canvas_length >
        config_.max_sequence_length)
        throw std::invalid_argument(
            "diffusion_gemma prompt plus complete output canvases exceeds built sequence capacity");
    reset();
    const auto prefill_start = std::chrono::steady_clock::now();
    encode(prompt);
    TextResult result;
    result.prefill_ms = elapsed_ms(prefill_start);
    const auto decode_start = std::chrono::steady_clock::now();
    std::mt19937_64 random(seed < 0 ? std::random_device{}() : static_cast<std::uint64_t>(seed));
    std::uniform_int_distribution<std::int32_t> random_token(0, config_.vocab_size - 1);
    for (std::int64_t block = 0; block < canvas_count; ++block) {
        std::vector<std::int32_t> canvas(static_cast<std::size_t>(config_.canvas_length));
        for (auto& token : canvas)
            token = random_token(random);
        auto conditioning = zero_conditioning(canvas.size());
        std::vector<std::vector<std::int32_t>> history;
        std::vector<std::int32_t> final_canvas;
        for (std::int32_t step = max_steps; step > 0; --step) {
            const float temperature = config_.t_min + (config_.t_max - config_.t_min) *
                                                          static_cast<float>(step) /
                                                          static_cast<float>(max_steps);
            auto observation = denoise(canvas, conditioning, temperature);
            auto sample =
                sample_canvas(reinterpret_cast<const float*>(observation.logits.bytes.data()),
                              canvas.size(), static_cast<std::size_t>(config_.vocab_size),
                              temperature, config_.entropy_bound, random);
            const bool stable =
                config_.stability_threshold == 0 ||
                (history.size() >= static_cast<std::size_t>(config_.stability_threshold) &&
                 std::all_of(history.end() - config_.stability_threshold, history.end(),
                             [&](const auto& prior) { return prior == sample.argmax; }));
            final_canvas = std::move(sample.argmax);
            history.push_back(final_canvas);
            if (history.size() > static_cast<std::size_t>(config_.stability_threshold))
                history.erase(history.begin());
            conditioning = std::move(observation.self_conditioning);
            canvas = std::move(sample.next_canvas);
            if (stable && sample.mean_entropy < config_.confidence_threshold)
                break;
        }
        const auto eos = std::find_if(final_canvas.begin(), final_canvas.end(), [&](const auto id) {
            return std::find(config_.eos_token_ids.begin(), config_.eos_token_ids.end(), id) !=
                   config_.eos_token_ids.end();
        });
        const auto end = eos == final_canvas.end() ? eos : eos + 1;
        result.token_ids.insert(result.token_ids.end(), final_canvas.begin(), end);
        if (eos != final_canvas.end())
            break;
        if (block + 1 < canvas_count)
            encode_tokens(final_canvas, false);
    }
    result.decode_ms = elapsed_ms(decode_start);
    result.text = tokenizer_->decode(result.token_ids);
    return result;
}

internal::TextResult Pipeline::run(const internal::TextContinuationRequest& request,
                                   internal::ConfigView supplied) {
    internal::validate_config(kGenerationFields, supplied);
    const auto budget =
        internal::config_get<std::int64_t>(supplied, kGenerationFields, "max_new_tokens")
            .value_or(config_.max_new_tokens);
    const auto seed =
        internal::config_get<std::int64_t>(supplied, kGenerationFields, "seed").value_or(0);
    if (budget < 1 || budget > config_.max_sequence_length)
        throw internal::ConfigError("diffusion_gemma max_new_tokens exceeds the built capacity");
    std::vector<std::int32_t> ids;
    if (const auto* text = std::get_if<std::string_view>(&request.prefix)) {
        ids = tokenizer_->encode(std::string(*text));
    } else {
        const auto tokens = std::get<Span<const std::int32_t>>(request.prefix);
        ids.assign(tokens.begin(), tokens.end());
    }
    return generate_ids(ids, static_cast<std::int32_t>(budget), seed, config_.max_denoising_steps);
}

TextResult Pipeline::generate(const std::string& prompt, const TextGenerationConfig& supplied) {
    check_legacy_options(supplied);
    std::string prefix = prompt;
    if (supplied.use_chat_template) {
        prefix = "<bos>";
        if (supplied.enable_thinking || !supplied.system_prompt.empty()) {
            prefix += "<|turn>system\n";
            if (supplied.enable_thinking)
                prefix += "<|think|>\n";
            prefix += trim(supplied.system_prompt) + "<turn|>\n";
        }
        prefix += "<|turn>user\n" + trim(prompt) + "<turn|>\n<|turn>model\n";
    }
    auto ids = tokenizer_->encode(prefix);
    return generate_ids(ids, supplied.max_new_tokens, supplied.seed,
                        supplied.num_steps == -1 ? config_.max_denoising_steps
                                                 : supplied.num_steps);
}

} // namespace trtmc::diffusion_gemma
