/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace trtmc::sam2::test {

struct GoldenBbox {
    std::array<float, 4> model_xyxy{};
    std::array<float, 4> original_xyxy{};
    float score{0.0F};
    std::int32_t label{-1};
};

struct GoldenFixture {
    GoldenBbox bbox;
    std::vector<std::uint8_t> masks;
    std::array<std::uint64_t, 5> foreground_pixels{};
};

struct MaskAccuracy {
    std::array<double, 5> frame_iou{};
    double macro_iou{0.0};
    double global_iou{0.0};

    bool passes() const;
};

struct BboxAccuracy {
    double iou{0.0};
    double max_coordinate_error{0.0};
    double score_error{0.0};
    bool label_exact{false};

    bool passes() const;
};

GoldenFixture loadGoldenFixture(const std::filesystem::path& directory);
MaskAccuracy compareMasks(const std::vector<std::uint8_t>& candidate,
                          const GoldenFixture& reference);
BboxAccuracy compareBbox(const GoldenBbox& candidate, const GoldenFixture& reference);

} // namespace trtmc::sam2::test
