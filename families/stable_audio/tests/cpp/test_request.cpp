/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU checks of the request side: the Task's config fields, their defaults, the checks
// StableAudioPipeline.check_inputs makes, the returned sample window, and the T5 token
// framing (max_length padding with truncation that keeps </s>).

#include "families/stable_audio/runtime/request.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace {

using trtmc::internal::ConfigEntry;
using trtmc::internal::ConfigError;
using trtmc::internal::ConfigKind;
using trtmc::internal::ConfigValue;
using trtmc::internal::ConfigView;
using trtmc::internal::TextToAudioRequest;
using trtmc::stable_audio::frame_tokens;
using trtmc::stable_audio::GenerationPlan;
using trtmc::stable_audio::plan_request;
using trtmc::stable_audio::StableAudioConfig;

int failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "failed: %s\n", what);
        ++failures;
    }
}

// The published checkpoint: 1,024 latent frames x 2,048 samples at 44.1 kHz.
StableAudioConfig published() {
    StableAudioConfig config;
    config.audio_samples = 2097152;
    return config;
}

GenerationPlan plan(const std::vector<ConfigEntry>& entries) {
    return plan_request(published(), TextToAudioRequest{"a prompt"},
                        ConfigView{entries.data(), entries.size()});
}

void expect_rejected(const std::vector<ConfigEntry>& entries, const char* what) {
    try {
        plan(entries);
    } catch (const ConfigError&) {
        return;
    }
    std::fprintf(stderr, "not rejected: %s\n", what);
    ++failures;
}

void check_fields() {
    const auto fields = trtmc::stable_audio::config_fields();
    expect(fields.size() == 6, "six config fields");
    const char* names[] = {"negative_prompt",     "audio_start_in_s", "audio_end_in_s",
                           "num_inference_steps", "guidance_scale",   "seed"};
    const ConfigKind kinds[] = {ConfigKind::String, ConfigKind::F64, ConfigKind::F64,
                                ConfigKind::I64,    ConfigKind::F64, ConfigKind::I64};
    for (std::size_t i = 0; i < fields.size() && i < 6; ++i) {
        expect(fields[i].name == names[i], "field order and names");
        expect(fields[i].kind == kinds[i], "field kinds");
    }
    // Fixed defaults only where they do not depend on the bundle.
    expect(!fields[0].default_value && fields[1].default_value && !fields[2].default_value &&
               !fields[3].default_value && !fields[4].default_value && fields[5].default_value,
           "engaged defaults: audio_start_in_s and seed only");
}

void check_defaults() {
    const auto defaults = plan({});
    expect(defaults.prompt == "a prompt", "prompt copied");
    expect(!defaults.negative_prompt, "no negative prompt by default");
    expect(defaults.num_inference_steps == 100, "default steps 100");
    expect(defaults.guidance_scale == 7.0, "default guidance 7.0");
    expect(defaults.seed == 0, "default seed 0");
    expect(defaults.audio_start_in_s == 0.0, "default start 0");
    // 1024 * 2048 / 44100 s, the whole window: [0, 2097152).
    expect(defaults.audio_end_in_s == 1024.0 * 2048.0 / 44100.0, "default end is the window");
    expect(defaults.first_sample == 0 && defaults.last_sample == 2097152, "whole window");
}

void check_overrides() {
    const auto window = plan({{"audio_start_in_s", ConfigValue{2.0}},
                              {"audio_end_in_s", ConfigValue{7.5}},
                              {"num_inference_steps", ConfigValue{std::int64_t{30}}},
                              {"guidance_scale", ConfigValue{1.0}},
                              {"seed", ConfigValue{std::int64_t{2}}},
                              {"negative_prompt", ConfigValue{std::string_view{"Low quality."}}}});
    expect(window.first_sample == 88200 && window.last_sample == 330750,
           "[int(2*sr), int(7.5*sr))");
    expect(window.num_inference_steps == 30 && window.guidance_scale == 1.0 && window.seed == 2,
           "overrides applied");
    expect(window.negative_prompt && *window.negative_prompt == "Low quality.", "negative prompt");
    const auto largest = plan(
        {{"guidance_scale", ConfigValue{static_cast<double>(std::numeric_limits<float>::max())}}});
    expect(largest.guidance_scale == static_cast<double>(std::numeric_limits<float>::max()),
           "the largest float32 guidance is accepted");
    // An end past the window is clipped like a Python slice.
    const auto clipped =
        plan({{"audio_start_in_s", ConfigValue{40.0}}, {"audio_end_in_s", ConfigValue{50.0}}});
    expect(clipped.first_sample == 1764000 && clipped.last_sample == 2097152, "clip to the window");
}

void check_rejections() {
    expect_rejected({{"num_inference_steps", ConfigValue{std::int64_t{0}}}}, "zero steps");
    expect_rejected({{"num_inference_steps", ConfigValue{std::int64_t{1001}}}}, "too many steps");
    expect_rejected({{"seed", ConfigValue{std::int64_t{-1}}}}, "negative seed");
    expect_rejected({{"guidance_scale", ConfigValue{std::numeric_limits<double>::quiet_NaN()}}},
                    "NaN guidance");
    expect_rejected({{"guidance_scale", ConfigValue{1e40}}}, "guidance beyond the float32 range");
    expect_rejected({{"guidance_scale", ConfigValue{-1e40}}}, "guidance below the float32 range");
    expect_rejected({{"audio_start_in_s", ConfigValue{5.0}}, {"audio_end_in_s", ConfigValue{5.0}}},
                    "empty request");
    expect_rejected({{"audio_start_in_s", ConfigValue{6.0}}, {"audio_end_in_s", ConfigValue{5.0}}},
                    "end before start");
    expect_rejected({{"audio_end_in_s", ConfigValue{48.0}}}, "longer than the window");
    expect_rejected({{"audio_start_in_s", ConfigValue{-1.0}}, {"audio_end_in_s", ConfigValue{1.0}}},
                    "start below the conditioning range");
    expect_rejected(
        {{"audio_start_in_s", ConfigValue{500.0}}, {"audio_end_in_s", ConfigValue{513.0}}},
        "end above the conditioning range");
    expect_rejected(
        {{"audio_start_in_s", ConfigValue{50.0}}, {"audio_end_in_s", ConfigValue{60.0}}},
        "window beyond the generated audio");
    expect_rejected({{"seed", ConfigValue{1.0}}}, "seed with the wrong type");
    expect_rejected({{"steps", ConfigValue{std::int64_t{4}}}}, "unknown field");
    expect_rejected(
        {{"seed", ConfigValue{std::int64_t{1}}}, {"seed", ConfigValue{std::int64_t{2}}}},
        "duplicate field");
}

void check_token_framing() {
    const auto config = published();
    std::vector<std::int32_t> ids;
    std::vector<float> mask;
    frame_tokens({37, 1345, 13}, config, ids, mask);
    expect(ids.size() == 128 && mask.size() == 128, "padded to 128");
    expect(ids[0] == 37 && ids[2] == 13 && ids[3] == 1 && ids[4] == 0, "pieces, </s>, padding");
    expect(mask[3] == 1.0F && mask[4] == 0.0F, "mask covers </s> only up to it");
    frame_tokens({}, config, ids, mask);
    expect(ids[0] == 1 && mask[0] == 1.0F && mask[1] == 0.0F, "empty prompt is </s>");
    frame_tokens(std::vector<std::int32_t>(300, 5), config, ids, mask);
    expect(ids[126] == 5 && ids[127] == 1 && mask[127] == 1.0F, "truncated to 127 pieces + </s>");
}

} // namespace

int main() {
    check_fields();
    check_defaults();
    check_overrides();
    check_rejections();
    check_token_framing();
    if (failures != 0) {
        std::fprintf(stderr, "stable_audio request: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("stable_audio request: config, window and token framing match the pipeline\n");
    return EXIT_SUCCESS;
}
