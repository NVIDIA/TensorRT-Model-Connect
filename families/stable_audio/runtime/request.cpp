/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_audio/runtime/request.h"

#include <algorithm>
#include <cmath>
#include <string_view>

namespace trtmc::stable_audio {
namespace {

using trtmc::internal::config_get;
using trtmc::internal::ConfigError;
using trtmc::internal::ConfigField;
using trtmc::internal::ConfigKind;
using trtmc::internal::ConfigValue;

constexpr std::int64_t kMaxSteps = 1000;

const ConfigField kConfigFields[] = {
    {"negative_prompt", ConfigKind::String, std::nullopt,
     "Prompt of the unconditional guidance branch; without it that branch is all zeros"},
    {"audio_start_in_s", ConfigKind::F64, ConfigValue{0.0},
     "Start of the returned audio in seconds (also a conditioning input)"},
    {"audio_end_in_s", ConfigKind::F64, std::nullopt,
     "End of the returned audio in seconds; default: the model's whole window"},
    {"num_inference_steps", ConfigKind::I64, std::nullopt, "Denoising steps; default 100"},
    {"guidance_scale", ConfigKind::F64, std::nullopt,
     "Classifier-free guidance scale; default 7.0, guidance is off at 1.0 or below"},
    {"seed", ConfigKind::I64, ConfigValue{std::int64_t{0}},
     "Seed of the family's counter-based noise generator"},
};

} // namespace

Span<const ConfigField> config_fields() noexcept {
    return kConfigFields;
}

double max_audio_length_in_s(const StableAudioConfig& config) noexcept {
    return static_cast<double>(config.latent_length) * static_cast<double>(config.hop_length) /
           static_cast<double>(config.sample_rate);
}

GenerationPlan plan_request(const StableAudioConfig& config,
                            const trtmc::internal::TextToAudioRequest& request,
                            trtmc::internal::ConfigView supplied) {
    trtmc::internal::validate_config(kConfigFields, supplied);
    GenerationPlan plan;
    plan.prompt = std::string(request.prompt);
    if (const auto negative =
            config_get<std::string_view>(supplied, kConfigFields, "negative_prompt"))
        plan.negative_prompt = std::string(*negative);
    plan.audio_start_in_s = *config_get<double>(supplied, kConfigFields, "audio_start_in_s");
    plan.audio_end_in_s = config_get<double>(supplied, kConfigFields, "audio_end_in_s")
                              .value_or(max_audio_length_in_s(config));
    const auto steps = config_get<std::int64_t>(supplied, kConfigFields, "num_inference_steps")
                           .value_or(config.default_num_inference_steps);
    if (steps < 1 || steps > kMaxSteps)
        throw ConfigError("stable_audio: num_inference_steps must be in [1, 1000]");
    plan.num_inference_steps = static_cast<std::int32_t>(steps);
    plan.guidance_scale = config_get<double>(supplied, kConfigFields, "guidance_scale")
                              .value_or(config.default_guidance_scale);
    if (!std::isfinite(plan.guidance_scale))
        throw ConfigError("stable_audio: guidance_scale must be finite");
    const auto seed = *config_get<std::int64_t>(supplied, kConfigFields, "seed");
    if (seed < 0)
        throw ConfigError("stable_audio: seed must be non-negative");
    plan.seed = static_cast<std::uint64_t>(seed);

    const double start = plan.audio_start_in_s;
    const double stop = plan.audio_end_in_s;
    if (!std::isfinite(start) || !std::isfinite(stop))
        throw ConfigError("stable_audio: audio_start_in_s and audio_end_in_s must be finite");
    // StableAudioPipeline.check_inputs: both inside the duration conditioners' range.
    if (start < config.duration_min || start > config.duration_max || stop < config.duration_min ||
        stop > config.duration_max)
        throw ConfigError(
            "stable_audio: audio_start_in_s / audio_end_in_s outside the conditioning range");
    if (!(stop > start))
        throw ConfigError("stable_audio: audio_end_in_s must exceed audio_start_in_s");
    if (stop - start > max_audio_length_in_s(config))
        throw ConfigError("stable_audio: the requested audio is longer than the model's window");
    // The whole window is denoised; Python slices [int(start * sr), int(end * sr)) of it.
    plan.first_sample = static_cast<std::int64_t>(start * config.sample_rate);
    plan.last_sample =
        std::min(static_cast<std::int64_t>(stop * config.sample_rate), config.audio_samples);
    if (plan.first_sample >= plan.last_sample)
        throw ConfigError("stable_audio: the requested window lies outside the generated audio");
    return plan;
}

void frame_tokens(std::vector<std::int32_t> pieces, const StableAudioConfig& config,
                  std::vector<std::int32_t>& ids, std::vector<float>& mask) {
    const auto limit = static_cast<std::size_t>(config.text_length - 1);
    if (pieces.size() > limit)
        pieces.resize(limit);
    pieces.push_back(config.eos_token_id);
    ids.assign(static_cast<std::size_t>(config.text_length), config.pad_token_id);
    mask.assign(static_cast<std::size_t>(config.text_length), 0.0F);
    std::copy(pieces.begin(), pieces.end(), ids.begin());
    std::fill_n(mask.begin(), pieces.size(), 1.0F);
}

} // namespace trtmc::stable_audio
