/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/cosmos3/options.h"

#include "trtmc/pipeline.h"

#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <tuple>

namespace trtmc {
namespace {

bool matches_qualified_profile(const Cosmos3Options& profile) {
    return std::tie(profile.video_height, profile.video_width, profile.video_num_frames,
                    profile.frame_rate, profile.num_inference_steps, profile.guidance_scale,
                    profile.flow_shift, profile.text_seq_len) ==
           std::tie(kCosmos3VideoHeight, kCosmos3VideoWidth, kCosmos3VideoFrames, kCosmos3FrameRate,
                    kCosmos3InferenceSteps, kCosmos3GuidanceScale, kCosmos3FlowShift,
                    kCosmos3TextSequenceLength);
}

void validate_profile(const Cosmos3Options& profile, const char* subject) {
    if (!matches_qualified_profile(profile)) {
        throw std::invalid_argument(std::string("Cosmos3-Nano ") + subject +
                                    " requires 1280x720, 189 frames, 24 FPS, 35 steps, CFG 6, "
                                    "flow shift 10, and a 4096-token profile");
    }
    if (profile.seed < 0)
        throw std::invalid_argument("Cosmos3-Nano seed must be non-negative");
    if (profile.negative_prompt.empty())
        throw std::invalid_argument("Cosmos3-Nano requires a non-empty negative prompt");
}

void validate_overrides(const GenerateConfig& config) {
    if (!config.initial_latents.empty())
        throw std::invalid_argument("Cosmos3-Nano does not support --initial-latents-raw");
    if (config.num_steps != -1 && config.num_steps <= 0)
        throw std::invalid_argument("Cosmos3-Nano num_steps must be -1 or positive");
    if (!std::isfinite(config.guidance_scale) ||
        (config.guidance_scale < 0.0F && config.guidance_scale != -1.0F)) {
        throw std::invalid_argument(
            "Cosmos3-Nano guidance_scale must be -1 or finite and non-negative");
    }
    if (config.seed < -1)
        throw std::invalid_argument("Cosmos3-Nano seed must be -1 or non-negative");
}

} // namespace

Cosmos3Options parse_cosmos3_options(const std::string& config_json) {
    Cosmos3Options options;
    const auto parsed = nlohmann::json::parse(config_json);
    options.negative_prompt = parsed.value("negative_prompt", std::string{});
    options.num_inference_steps = parsed.value("num_inference_steps", options.num_inference_steps);
    options.guidance_scale = parsed.value("guidance_scale", options.guidance_scale);
    options.flow_shift = parsed.value("flow_shift", options.flow_shift);
    options.seed = parsed.value("seed", options.seed);
    options.video_height = parsed.value("video_height", options.video_height);
    options.video_width = parsed.value("video_width", options.video_width);
    options.video_num_frames = parsed.value("video_num_frames", options.video_num_frames);
    options.frame_rate = parsed.value("frame_rate", options.frame_rate);
    options.text_seq_len = parsed.value("text_seq_len", options.text_seq_len);
    validate_profile(options, "bundle");
    return options;
}

Cosmos3Request resolve_cosmos3_request(const Cosmos3Options& options,
                                       const GenerateConfig& config) {
    validate_profile(options, "bundle");
    validate_overrides(config);
    Cosmos3Request request = options;
    request.negative_prompt =
        config.negative_prompt.empty() ? options.negative_prompt : config.negative_prompt;
    request.num_inference_steps =
        config.num_steps > 0 ? config.num_steps : options.num_inference_steps;
    request.guidance_scale =
        config.guidance_scale >= 0.0F ? config.guidance_scale : options.guidance_scale;
    request.seed = config.seed >= 0 ? config.seed : options.seed;
    if (config.height > 0 && config.height != options.video_height)
        throw std::invalid_argument("Cosmos3-Nano height must match the bundle profile");
    if (config.width > 0 && config.width != options.video_width)
        throw std::invalid_argument("Cosmos3-Nano width must match the bundle profile");
    validate_profile(request, "request");
    return request;
}

} // namespace trtmc
