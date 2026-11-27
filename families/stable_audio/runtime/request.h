/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// The request side of the stable_audio Task: the bundle facts the runtime reads, the
// Task's config fields, their defaults and checks, the returned sample window and the T5
// token framing. Free of engines and CUDA so the family's CPU tests can exercise it.

#include "families/stable_audio/runtime/scheduler.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/config.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace trtmc::stable_audio {

// The facts the runtime reads from runtime.json; names match the keys model.py writes.
struct StableAudioConfig {
    std::int32_t sample_rate{44100};
    std::int32_t audio_channels{2};
    std::int64_t audio_samples{0};
    std::int32_t hop_length{2048};
    std::int32_t latent_channels{64};
    std::int32_t latent_length{1024};
    std::int32_t text_length{128};
    std::int32_t pad_token_id{0};
    std::int32_t eos_token_id{1};
    std::int32_t encoder_tokens{130};
    std::int32_t conditioning_dim{768};
    std::int32_t global_dim{1536};
    double duration_min{0.0};
    double duration_max{512.0};
    std::int32_t default_num_inference_steps{100};
    double default_guidance_scale{7.0};
    SchedulerConfig scheduler;
};

// One request after defaults and range checks.
struct GenerationPlan {
    std::string prompt;
    std::optional<std::string> negative_prompt;
    double audio_start_in_s{0.0};
    double audio_end_in_s{0.0};
    std::int32_t num_inference_steps{0};
    double guidance_scale{0.0};
    std::uint64_t seed{0};
    // The returned samples: [first_sample, last_sample) of the decoded window.
    std::int64_t first_sample{0};
    std::int64_t last_sample{0};
};

// The Task's config field table (static storage).
Span<const trtmc::internal::ConfigField> config_fields() noexcept;

// sample_size * hop_length / sampling_rate in double, as StableAudioPipeline computes it.
double max_audio_length_in_s(const StableAudioConfig& config) noexcept;

// Resolve defaults and reject what the pipeline cannot honor (ConfigError).
GenerationPlan plan_request(const StableAudioConfig& config,
                            const trtmc::internal::TextToAudioRequest& request,
                            trtmc::internal::ConfigView supplied);

// padding="max_length", truncation=True: the first text_length - 1 pieces, then </s>, then
// padding; `mask` is 1 for every kept token (</s> included) and 0 for padding.
void frame_tokens(std::vector<std::int32_t> pieces, const StableAudioConfig& config,
                  std::vector<std::int32_t>& ids, std::vector<float>& mask);

} // namespace trtmc::stable_audio
