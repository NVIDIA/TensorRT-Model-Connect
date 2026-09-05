/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/cosmos3/runtime/runtime_config.h"

#include <array>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace trtmc::cosmos3 {
namespace {

template <typename T>
T require_value(const nlohmann::json& document, const char* name) {
    if (!document.contains(name))
        throw std::runtime_error(std::string("Cosmos3 runtime.json missing '") + name + "'");
    try {
        return document.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("Cosmos3 runtime.json has invalid '") + name + "'");
    }
}

void validate_fixed_profile(const RuntimeConfig& config) {
    if (config.negative_prompt.empty())
        throw std::runtime_error("Cosmos3 runtime.json requires a non-empty negative_prompt");
    if (config.num_inference_steps != kInferenceSteps || config.guidance_scale != kGuidanceScale ||
        config.flow_shift != kFlowShift || config.video_height != kVideoHeight ||
        config.video_width != kVideoWidth || config.video_num_frames != kVideoFrames ||
        config.frame_rate != kFrameRate || config.text_seq_len != kTextSequenceLength) {
        throw std::runtime_error(
            "Cosmos3 runtime.json must use the qualified 1280x720, 189-frame, 24 FPS, "
            "35-step, guidance 6, flow-shift 10, 4096-token profile");
    }
    if (config.seed < 0)
        throw std::runtime_error("Cosmos3 runtime.json seed must be non-negative");
    if (config.context_parallel_size != 1 && config.context_parallel_size != 2)
        throw std::runtime_error("Cosmos3 context_parallel_size must be 1 or 2");
}

} // namespace

RuntimeConfig parse_runtime_config(std::string_view text) {
    nlohmann::json document;
    try {
        document = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("Cosmos3 runtime.json is invalid JSON: " +
                                 std::string(error.what()));
    }
    if (!document.is_object())
        throw std::runtime_error("Cosmos3 runtime.json must be an object");
    constexpr std::array<const char*, 11> fields = {
        "negative_prompt",
        "num_inference_steps",
        "guidance_scale",
        "flow_shift",
        "seed",
        "video_height",
        "video_width",
        "video_num_frames",
        "frame_rate",
        "text_seq_len",
        "context_parallel_size",
    };
    if (document.size() != fields.size())
        throw std::runtime_error("Cosmos3 runtime.json has an unexpected field set");
    for (const char* field : fields) {
        if (!document.contains(field))
            throw std::runtime_error(std::string("Cosmos3 runtime.json missing '") + field + "'");
    }

    RuntimeConfig config{
        require_value<std::string>(document, "negative_prompt"),
        require_value<std::int32_t>(document, "num_inference_steps"),
        require_value<float>(document, "guidance_scale"),
        require_value<float>(document, "flow_shift"),
        require_value<std::int32_t>(document, "seed"),
        require_value<std::int32_t>(document, "video_height"),
        require_value<std::int32_t>(document, "video_width"),
        require_value<std::int32_t>(document, "video_num_frames"),
        require_value<std::int32_t>(document, "frame_rate"),
        require_value<std::int32_t>(document, "text_seq_len"),
        require_value<std::int32_t>(document, "context_parallel_size"),
    };
    validate_fixed_profile(config);
    return config;
}

GenerationRequest resolve_request(const RuntimeConfig& runtime,
                                  const ImageGenerationConfig& request) {
    validate_fixed_profile(runtime);
    if (request.num_samples != 1)
        throw std::invalid_argument("Cosmos3 generates exactly one video");
    if (request.cfg_scale != -1.0F || request.sde_gamma != -1.0F ||
        !request.initial_latents.empty()) {
        throw std::invalid_argument("Cosmos3 received unsupported image-generation options");
    }
    if (request.num_steps != -1 && request.num_steps != runtime.num_inference_steps)
        throw std::invalid_argument("Cosmos3 num_steps must match runtime.json");
    if (!std::isfinite(request.guidance_scale) ||
        (request.guidance_scale != -1.0F && request.guidance_scale != runtime.guidance_scale)) {
        throw std::invalid_argument("Cosmos3 guidance_scale must match runtime.json");
    }
    if (request.seed < -1)
        throw std::invalid_argument("Cosmos3 seed must be -1 or non-negative");
    if (request.height != 0 && request.height != runtime.video_height)
        throw std::invalid_argument("Cosmos3 height must match runtime.json");
    if (request.width != 0 && request.width != runtime.video_width)
        throw std::invalid_argument("Cosmos3 width must match runtime.json");

    const std::string negative_prompt =
        request.negative_prompt.empty() ? runtime.negative_prompt : request.negative_prompt;
    if (negative_prompt.empty())
        throw std::invalid_argument("Cosmos3 negative prompt must not be empty");
    return {
        negative_prompt,
        runtime.num_inference_steps,
        runtime.guidance_scale,
        runtime.flow_shift,
        request.seed == -1 ? runtime.seed : request.seed,
    };
}

} // namespace trtmc::cosmos3
