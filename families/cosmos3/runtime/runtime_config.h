/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/task.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace trtmc::cosmos3 {

inline constexpr std::int32_t kVideoHeight = 720;
inline constexpr std::int32_t kVideoWidth = 1280;
inline constexpr std::int32_t kVideoFrames = 189;
inline constexpr std::int32_t kFrameRate = 24;
inline constexpr std::int32_t kInferenceSteps = 35;
inline constexpr float kGuidanceScale = 6.0F;
inline constexpr float kFlowShift = 10.0F;
inline constexpr std::int32_t kTextSequenceLength = 4096;

struct RuntimeConfig {
    std::string negative_prompt;
    std::int32_t num_inference_steps;
    float guidance_scale;
    float flow_shift;
    std::int32_t seed;
    std::int32_t video_height;
    std::int32_t video_width;
    std::int32_t video_num_frames;
    std::int32_t frame_rate;
    std::int32_t text_seq_len;
    std::int32_t context_parallel_size;
};

struct GenerationRequest {
    std::string negative_prompt;
    std::int32_t num_inference_steps;
    float guidance_scale;
    float flow_shift;
    std::int32_t seed;
};

RuntimeConfig parse_runtime_config(std::string_view text);
GenerationRequest resolve_request(const RuntimeConfig& runtime,
                                  const ImageGenerationConfig& request);

} // namespace trtmc::cosmos3
