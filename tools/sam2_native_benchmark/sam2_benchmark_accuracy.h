/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam2/sam2_video_session.h"
#include "sam2_benchmark_protocol.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace trtmc::sam2::benchmark {

struct GoldenEvidence {
    std::array<float, 4> bbox_original_xyxy{};
    float bbox_score{0.0F};
    std::int32_t bbox_label{-1};
    std::vector<std::uint8_t> masks;
    std::array<std::uint64_t, 5> foreground_pixels{};
};

GoldenEvidence loadGoldenEvidence(const std::filesystem::path& directory);

// Explicitly materializes all five device masks and computes every accuracy
// gate. Call only in an untimed qualification replay.
AccuracyReplay evaluateAccuracy(std::size_t replay_index, const Sam2VideoPromptResult& prompt,
                                Sam2VideoFrameResults& results, const GoldenEvidence& golden);

} // namespace trtmc::sam2::benchmark
