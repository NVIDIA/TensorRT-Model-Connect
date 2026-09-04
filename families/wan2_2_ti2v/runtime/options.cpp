/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wan2_2_ti2v/runtime/options.h"

#include "trtmc/task.h"

#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

void validate_common_profile(const Wan22TI2VOptions& profile) {
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
}

bool is_official_profile(const Wan22TI2VOptions& profile) {
    return profile.num_inference_steps == kWan22OfficialInferenceSteps &&
           profile.video_height == kWan22OfficialVideoHeight &&
           profile.video_width == kWan22OfficialVideoWidth &&
           profile.video_num_frames == kWan22OfficialVideoFrames;
}

bool is_l0_profile(const Wan22TI2VOptions& profile) {
    return profile.num_inference_steps == kWan22L0InferenceSteps &&
           profile.video_height == kWan22L0VideoHeight &&
           profile.video_width == kWan22L0VideoWidth &&
           profile.video_num_frames == kWan22L0VideoFrames;
}

void validate_profile(const Wan22TI2VOptions& profile, const char* subject) {
    validate_common_profile(profile);
    if (is_official_profile(profile) || is_l0_profile(profile))
        return;

    throw std::invalid_argument(
        std::string("Wan2.2-TI2V-5B ") + subject +
        " requires one complete qualified profile: 1280x704/121 frames/50 steps or "
        "672x384/5 frames/15 steps");
}

void validate_request_overrides(const ImageGenerationConfig& config) {
    if (!config.initial_latents.empty()) {
        throw std::invalid_argument("Wan2.2-TI2V-5B does not support --initial-latents-raw");
    }
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

void validate_resolved_request(const Wan22TI2VOptions& options, const Wan22TI2VRequest& request,
                               const ImageGenerationConfig& config) {
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
}

} // namespace

Wan22TI2VOptions parse_wan22_options(const std::string& config_json) {
    Wan22TI2VOptions options;
    const auto parsed = nlohmann::json::parse(config_json);
    options.negative_prompt = parsed.at("negative_prompt").get<std::string>();
    options.num_inference_steps = parsed.at("num_inference_steps").get<std::int32_t>();
    options.guidance_scale = parsed.at("guidance_scale").get<float>();
    options.flow_shift = parsed.at("flow_shift").get<float>();
    options.seed = parsed.at("seed").get<std::int32_t>();
    options.video_height = parsed.at("video_height").get<std::int32_t>();
    options.video_width = parsed.at("video_width").get<std::int32_t>();
    options.video_num_frames = parsed.at("video_num_frames").get<std::int32_t>();
    options.frame_rate = parsed.at("frame_rate").get<std::int32_t>();
    options.text_seq_len = parsed.at("text_seq_len").get<std::int32_t>();
    if (options.negative_prompt.empty())
        throw std::runtime_error("Wan2.2 bundle config is missing the official negative prompt");
    validate_profile(options, "bundle");
    return options;
}

Wan22TI2VRequest resolve_wan22_request(const Wan22TI2VOptions& options,
                                       const ImageGenerationConfig& config) {
    validate_profile(options, "bundle");
    validate_request_overrides(config);
    Wan22TI2VRequest request = options;
    request.negative_prompt =
        config.negative_prompt.empty() ? options.negative_prompt : config.negative_prompt;
    request.num_inference_steps =
        config.num_steps > 0 ? config.num_steps : options.num_inference_steps;
    request.guidance_scale =
        config.guidance_scale >= 0.0F ? config.guidance_scale : options.guidance_scale;
    request.seed = config.seed >= 0 ? config.seed : options.seed;
    validate_resolved_request(options, request, config);
    return request;
}

} // namespace trtmc
