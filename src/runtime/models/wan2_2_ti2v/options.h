/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc {

struct GenerateConfig;

inline constexpr int32_t kWan22OfficialVideoHeight = 704;
inline constexpr int32_t kWan22OfficialVideoWidth = 1280;
inline constexpr int32_t kWan22OfficialVideoFrames = 121;
inline constexpr int32_t kWan22OfficialFrameRate = 24;
inline constexpr int32_t kWan22OfficialInferenceSteps = 50;
inline constexpr float kWan22OfficialGuidanceScale = 5.0F;
inline constexpr float kWan22OfficialFlowShift = 5.0F;

struct Wan22TI2VOptions {
    std::string negative_prompt;
    int32_t num_inference_steps{kWan22OfficialInferenceSteps};
    float guidance_scale{kWan22OfficialGuidanceScale};
    float flow_shift{kWan22OfficialFlowShift};
    int32_t seed{42};
    int32_t video_height{kWan22OfficialVideoHeight};
    int32_t video_width{kWan22OfficialVideoWidth};
    int32_t video_num_frames{kWan22OfficialVideoFrames};
    int32_t frame_rate{kWan22OfficialFrameRate};
};

// Fully resolved request consumed by the fixed-shape native runtime. Keeping
// this resolution in one place prevents CLI overrides from being parsed but
// then silently ignored by the Wan pipeline.
struct Wan22TI2VRequest {
    std::string negative_prompt;
    int32_t num_inference_steps{kWan22OfficialInferenceSteps};
    float guidance_scale{kWan22OfficialGuidanceScale};
    float flow_shift{kWan22OfficialFlowShift};
    int32_t seed{42};
    int32_t video_height{kWan22OfficialVideoHeight};
    int32_t video_width{kWan22OfficialVideoWidth};
    int32_t video_num_frames{kWan22OfficialVideoFrames};
    int32_t frame_rate{kWan22OfficialFrameRate};
};

// Parse with a real JSON implementation so escaped Unicode reaches the native
// tokenizer as UTF-8 rather than literal backslash-u text.
Wan22TI2VOptions parse_wan22_options(const std::string& config_json);

// Resolve caller overrides and reject any request that cannot be honored by
// the static 1280x704, 121-frame TensorRT engines.
Wan22TI2VRequest resolve_wan22_request(const Wan22TI2VOptions& options,
                                       const GenerateConfig& config);

} // namespace trtmc
