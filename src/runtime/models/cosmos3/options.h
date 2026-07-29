/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc {

struct GenerateConfig;

inline constexpr int32_t kCosmos3VideoHeight = 720;
inline constexpr int32_t kCosmos3VideoWidth = 1280;
inline constexpr int32_t kCosmos3VideoFrames = 189;
inline constexpr int32_t kCosmos3FrameRate = 24;
inline constexpr int32_t kCosmos3InferenceSteps = 35;
inline constexpr float kCosmos3GuidanceScale = 6.0F;
inline constexpr float kCosmos3FlowShift = 10.0F;
inline constexpr int32_t kCosmos3TextSequenceLength = 4096;

struct Cosmos3Options {
    std::string negative_prompt;
    int32_t num_inference_steps{kCosmos3InferenceSteps};
    float guidance_scale{kCosmos3GuidanceScale};
    float flow_shift{kCosmos3FlowShift};
    int32_t seed{42};
    int32_t video_height{kCosmos3VideoHeight};
    int32_t video_width{kCosmos3VideoWidth};
    int32_t video_num_frames{kCosmos3VideoFrames};
    int32_t frame_rate{kCosmos3FrameRate};
    int32_t text_seq_len{kCosmos3TextSequenceLength};
};

using Cosmos3Request = Cosmos3Options;

Cosmos3Options parse_cosmos3_options(const std::string& config_json);
Cosmos3Request resolve_cosmos3_request(const Cosmos3Options& options, const GenerateConfig& config);

} // namespace trtmc
