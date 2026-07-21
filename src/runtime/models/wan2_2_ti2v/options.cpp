/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/options.h"

#include "trtmc/pipeline.h"

#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

template <typename Profile>
Wan22TI2VProfileKind require_profile(const Profile& profile, const char* subject) {
    if (profile.guidance_scale != kWan22OfficialGuidanceScale)
        throw std::invalid_argument("Wan2.2-TI2V-5B bundle requires CFG=5");
    if (profile.flow_shift != kWan22OfficialFlowShift)
        throw std::invalid_argument("Wan2.2-TI2V-5B bundle requires flow_shift=5");
    if (profile.frame_rate != kWan22OfficialFrameRate)
        throw std::invalid_argument("Wan2.2-TI2V-5B bundle requires frame_rate=24");
    if (profile.text_seq_len != kWan22TextSequenceLength)
        throw std::invalid_argument("Wan2.2-TI2V-5B bundle requires text_seq_len=512");
    if (profile.seed < 0)
        throw std::invalid_argument("Wan2.2-TI2V-5B bundle seed must be non-negative");

    const bool official = profile.num_inference_steps == kWan22OfficialInferenceSteps &&
                          profile.video_height == kWan22OfficialVideoHeight &&
                          profile.video_width == kWan22OfficialVideoWidth &&
                          profile.video_num_frames == kWan22OfficialVideoFrames;
    if (official)
        return Wan22TI2VProfileKind::kOfficial;

    const bool l0 = profile.num_inference_steps == kWan22L0InferenceSteps &&
                    profile.video_height == kWan22L0VideoHeight &&
                    profile.video_width == kWan22L0VideoWidth &&
                    profile.video_num_frames == kWan22L0VideoFrames;
    if (l0)
        return Wan22TI2VProfileKind::kL0;

    throw std::invalid_argument(
        std::string("Wan2.2-TI2V-5B ") + subject +
        " requires one complete qualified profile: 1280x704/121 frames/50 steps or "
        "672x384/5 frames/15 steps");
}

void validate_request_overrides(const GenerateConfig& config) {
    if (config.num_steps != -1 && config.num_steps <= 0)
        throw std::invalid_argument("Wan2.2-TI2V-5B num_steps must be -1 or a positive integer");
    if (!std::isfinite(config.guidance_scale) ||
        (config.guidance_scale < 0.0F && config.guidance_scale != -1.0F)) {
        throw std::invalid_argument(
            "Wan2.2-TI2V-5B guidance_scale must be -1 or a finite non-negative value");
    }
    if (config.seed < -1)
        throw std::invalid_argument("Wan2.2-TI2V-5B seed must be -1 or non-negative");
}

Wan22TI2VRequest make_wan22_request(const Wan22TI2VOptions& options, const GenerateConfig& config) {
    Wan22TI2VRequest request;
    request.negative_prompt =
        config.negative_prompt.empty() ? options.negative_prompt : config.negative_prompt;
    request.num_inference_steps =
        config.num_steps > 0 ? config.num_steps : options.num_inference_steps;
    request.guidance_scale =
        config.guidance_scale >= 0.0F ? config.guidance_scale : options.guidance_scale;
    request.flow_shift = options.flow_shift;
    request.seed = config.seed >= 0 ? config.seed : options.seed;
    request.video_height = options.video_height;
    request.video_width = options.video_width;
    request.video_num_frames = options.video_num_frames;
    request.frame_rate = options.frame_rate;
    request.text_seq_len = options.text_seq_len;
    return request;
}

void validate_resolved_request(const Wan22TI2VOptions& options, const Wan22TI2VRequest& request,
                               const GenerateConfig& config) {
    if (request.num_inference_steps != options.num_inference_steps) {
        throw std::invalid_argument(
            "Wan2.2-TI2V-5B --num-steps must match the bundle's complete profile");
    }
    if (request.guidance_scale != options.guidance_scale) {
        throw std::invalid_argument(
            "Wan2.2-TI2V-5B --guidance-scale must match the bundle's complete profile");
    }
    if (config.height != 0 && config.height != request.video_height) {
        throw std::invalid_argument("Wan2.2-TI2V-5B --height must match bundle profile height " +
                                    std::to_string(request.video_height));
    }
    if (config.width != 0 && config.width != request.video_width) {
        throw std::invalid_argument("Wan2.2-TI2V-5B --width must match bundle profile width " +
                                    std::to_string(request.video_width));
    }
    if (require_profile(request, "request") != require_profile(options, "bundle"))
        throw std::invalid_argument("Wan2.2-TI2V-5B request profile differs from its bundle");
}

} // namespace

Wan22TI2VOptions parse_wan22_options(const std::string& config_json) {
    Wan22TI2VOptions options;
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
    if (options.negative_prompt.empty())
        throw std::runtime_error("Wan2.2 bundle config is missing the official negative prompt");
    (void)require_wan22_profile(options);
    return options;
}

Wan22TI2VProfileKind require_wan22_profile(const Wan22TI2VOptions& options) {
    return require_profile(options, "bundle");
}

Wan22TI2VProfileKind require_wan22_profile(const Wan22TI2VRequest& request) {
    return require_profile(request, "request");
}

Wan22TI2VRequest resolve_wan22_request(const Wan22TI2VOptions& options,
                                       const GenerateConfig& config) {
    (void)require_wan22_profile(options);
    validate_request_overrides(config);
    auto request = make_wan22_request(options, config);
    validate_resolved_request(options, request, config);
    return request;
}

} // namespace trtmc
