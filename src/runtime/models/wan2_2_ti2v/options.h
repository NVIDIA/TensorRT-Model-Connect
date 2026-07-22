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
inline constexpr int32_t kWan22L0VideoHeight = 384;
inline constexpr int32_t kWan22L0VideoWidth = 672;
inline constexpr int32_t kWan22L0VideoFrames = 5;
inline constexpr int32_t kWan22L0InferenceSteps = 15;
inline constexpr int32_t kWan22TextSequenceLength = 512;

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
    int32_t text_seq_len{kWan22TextSequenceLength};
};

// A request has the same fixed-profile fields as the bundle options, with
// permitted caller overrides resolved before validation.
using Wan22TI2VRequest = Wan22TI2VOptions;

// Parse with a real JSON implementation so escaped Unicode reaches the native
// tokenizer as UTF-8 rather than literal backslash-u text.
Wan22TI2VOptions parse_wan22_options(const std::string& config_json);

// Resolve caller overrides and reject any request that cannot be honored by
// the profile-specific static TensorRT engines embedded in the bundle.
Wan22TI2VRequest resolve_wan22_request(const Wan22TI2VOptions& options,
                                       const GenerateConfig& config);

} // namespace trtmc
