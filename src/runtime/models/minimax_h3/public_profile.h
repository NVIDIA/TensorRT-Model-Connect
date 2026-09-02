/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace trtmc {

inline constexpr int32_t kMiniMaxH3DefaultOutputFrames = 124;
inline constexpr int32_t kMiniMaxH3DefaultOutputHeight = 768;
inline constexpr int32_t kMiniMaxH3DefaultOutputWidth = 1344;
inline constexpr int32_t kMiniMaxH3CanvasMultiple = 32;
inline constexpr int32_t kMiniMaxH3CanvasShortEdge = 768;
inline constexpr int64_t kMiniMaxH3CanvasMaxPixels = static_cast<int64_t>(768) * 1344;
inline constexpr int32_t kMiniMaxH3ExplicitCanvasHeight = 544;
inline constexpr int32_t kMiniMaxH3ExplicitCanvasWidth = 960;

struct MiniMaxH3Canvas {
    int32_t height{0};
    int32_t width{0};
};

namespace minimax_h3_profile_detail {

inline int32_t round_to_even_multiple(double value, int32_t multiple) {
    const double scaled = value / multiple;
    const double lower = std::floor(scaled);
    const double fraction = scaled - lower;
    double rounded = lower;
    if (fraction > 0.5 || (fraction == 0.5 && static_cast<int64_t>(lower) % 2 != 0))
        rounded += 1.0;
    if (rounded > static_cast<double>(std::numeric_limits<int32_t>::max() / multiple))
        throw std::overflow_error("MiniMax-H3 canvas rounding overflow");
    return std::max(multiple, static_cast<int32_t>(rounded) * multiple);
}

inline bool landscape_canvas_is_official(int32_t height, int32_t width) {
    if (height == kMiniMaxH3CanvasShortEdge && width >= kMiniMaxH3CanvasShortEdge &&
        width <= kMiniMaxH3DefaultOutputWidth) {
        return true;
    }
    if (height > kMiniMaxH3CanvasShortEdge || width <= kMiniMaxH3DefaultOutputWidth)
        return false;

    // Above 16:9, the pre-round axes lie on h*w=768*1344. Intersect
    // the two nearest-32 rounding bins with the trained ratio interval.
    const double lower = std::max(
        {static_cast<double>(height - kMiniMaxH3CanvasMultiple / 2),
         static_cast<double>(kMiniMaxH3CanvasMaxPixels) / (width + kMiniMaxH3CanvasMultiple / 2),
         std::sqrt(static_cast<double>(kMiniMaxH3CanvasMaxPixels) / 4.0)});
    const double upper = std::min(
        {static_cast<double>(height + kMiniMaxH3CanvasMultiple / 2),
         static_cast<double>(kMiniMaxH3CanvasMaxPixels) / (width - kMiniMaxH3CanvasMultiple / 2),
         static_cast<double>(kMiniMaxH3CanvasShortEdge)});
    return lower < upper;
}

} // namespace minimax_h3_profile_detail

inline bool is_minimax_h3_native_canvas(int32_t height, int32_t width) {
    if (height <= 0 || width <= 0 || height % kMiniMaxH3CanvasMultiple != 0 ||
        width % kMiniMaxH3CanvasMultiple != 0) {
        return false;
    }
    if ((height == kMiniMaxH3ExplicitCanvasHeight && width == kMiniMaxH3ExplicitCanvasWidth) ||
        (height == kMiniMaxH3ExplicitCanvasWidth && width == kMiniMaxH3ExplicitCanvasHeight)) {
        return true;
    }
    const double ratio = static_cast<double>(width) / height;
    if (ratio < 0.25 || ratio > 4.0)
        return false;
    return width >= height ? minimax_h3_profile_detail::landscape_canvas_is_official(height, width)
                           : minimax_h3_profile_detail::landscape_canvas_is_official(width, height);
}

inline MiniMaxH3Canvas resolve_minimax_h3_canvas(double aspect_width, double aspect_height) {
    if (!std::isfinite(aspect_width) || !std::isfinite(aspect_height) || aspect_width <= 0.0 ||
        aspect_height <= 0.0) {
        throw std::invalid_argument("MiniMax-H3 aspect ratio must be finite and positive");
    }
    const double ratio = aspect_width / aspect_height;
    if (!std::isfinite(ratio) || ratio < 0.25 || ratio > 4.0)
        throw std::invalid_argument("MiniMax-H3 output aspect ratio must be between 1:4 and 4:1");

    double width = 0.0;
    double height = 0.0;
    if (ratio >= 1.0) {
        width = kMiniMaxH3CanvasShortEdge * ratio;
        height = kMiniMaxH3CanvasShortEdge;
    } else {
        width = kMiniMaxH3CanvasShortEdge;
        height = kMiniMaxH3CanvasShortEdge / ratio;
    }
    const double area = width * height;
    if (area > static_cast<double>(kMiniMaxH3CanvasMaxPixels)) {
        const double scale = std::sqrt(static_cast<double>(kMiniMaxH3CanvasMaxPixels) / area);
        width *= scale;
        height *= scale;
    }

    MiniMaxH3Canvas result;
    result.height =
        minimax_h3_profile_detail::round_to_even_multiple(height, kMiniMaxH3CanvasMultiple);
    result.width =
        minimax_h3_profile_detail::round_to_even_multiple(width, kMiniMaxH3CanvasMultiple);
    if (!is_minimax_h3_native_canvas(result.height, result.width))
        throw std::logic_error("MiniMax-H3 canvas resolver produced an invalid canvas");
    return result;
}

inline int32_t align_minimax_h3_num_frames(int32_t requested_frames) {
    if (requested_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 requested frame count must be positive");
    int32_t aligned = requested_frames;
    while (aligned % 17 != 5) {
        if (aligned == std::numeric_limits<int32_t>::max())
            throw std::overflow_error("MiniMax-H3 frame alignment overflow");
        ++aligned;
    }
    if (aligned < 5 * 24 || aligned > 15 * 24)
        throw std::invalid_argument(
            "MiniMax-H3 released local profile supports aligned durations from 5 to 15 seconds");
    return aligned;
}

} // namespace trtmc
