/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/video.h"

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::videomae {

// One bundle's clip: `frames` RGB frames of `size` x `size` pixels, cut into
// tubes of `tubelet` frames x `patch` x `patch` pixels. Tube numbers run over
// (tubelet, row, column): tube = (t * grid + row) * grid + column.
struct ClipGeometry {
    static constexpr std::int64_t kChannels = 3;
    std::int64_t frames{0};
    std::int64_t size{0};
    std::int64_t patch{0};
    std::int64_t tubelet{0};

    std::int64_t grid() const noexcept { return size / patch; }
    std::int64_t tubes() const noexcept { return frames / tubelet * grid() * grid(); }
    std::int64_t tube_pixels() const noexcept { return tubelet * patch * patch; }
    std::int64_t tube_values() const noexcept { return tube_pixels() * kChannels; }
    std::int64_t frame_values() const noexcept { return size * size * kChannels; }
};

// Throws std::runtime_error unless the geometry describes whole tubes of a
// clip small enough for every index the runtime computes.
void validate_geometry(const ClipGeometry& geometry);

struct PixelNormalization {
    std::array<float, 3> mean{};
    std::array<float, 3> std{};
};

// The request frames as one THWC float32 clip in [0, 1]. 8-bit values are
// scaled as the checkpoint's processor rescales them: the double product with
// 1/255, rounded to float32. Float32 frames must already lie in [0, 1].
std::vector<float> source_clip(Span<const internal::ImageView> frames,
                               const ClipGeometry& geometry);

// The engine input `pixel_values`: [1, frames, 3, size, size], normalized per
// channel in float32 as (value - mean) / std.
std::vector<float> pixel_values(const std::vector<float>& clip, const ClipGeometry& geometry,
                                const PixelNormalization& normalization);

struct TubeSelection {
    std::vector<std::int32_t> visible; // Conditioning tubes, ascending.
    std::vector<std::int32_t> masked;  // Tubes to reconstruct, ascending.
};

// Reads a mask that selects whole tubes: every value is 0 (condition) or 1
// (reconstruct) and constant on each tube; at least one tube of each kind.
TubeSelection select_tubes(const internal::VideoMaskView& mask, const ClipGeometry& geometry);

// Replaces every masked tube of `clip` by its prediction. Row i of
// `predictions` holds tube selection.masked[i], normalized by that tube's own
// per-channel mean and unbiased standard deviation, values in (frame, row,
// column, channel) order. The statistics are taken from the source pixels of
// the tube; the result is clamped to [0, 1].
void reconstruct(std::vector<float>& clip, const float* predictions, const TubeSelection& selection,
                 const ClipGeometry& geometry);

} // namespace trtmc::videomae
