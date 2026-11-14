/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_groot/runtime/image.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc::lerobot_groot {
namespace {

constexpr std::int32_t kCoefBits = 11;
constexpr std::int32_t kCoefScale = 1 << kCoefBits;

struct Axis {
    std::vector<std::int32_t> index;  // Left (or top) source sample per output position.
    std::vector<std::int32_t> weight; // Two fixed-point weights per output position.
    std::int32_t limit{0};            // First output position that only replicates the border.
};

// cv::resize's coefficient setup in area mode: sx = floor(dx * scale) and the fractional part of
// (dx + 1) - (sx + 1) / scale, both from double arithmetic rounded to float as OpenCV does.
Axis area_axis(std::int32_t src, std::int32_t dst) {
    const double inv_scale = static_cast<double>(dst) / src;
    const double scale = 1.0 / inv_scale;
    Axis axis;
    axis.index.resize(static_cast<std::size_t>(dst));
    axis.weight.resize(static_cast<std::size_t>(dst) * 2);
    axis.limit = dst;
    for (std::int32_t d = 0; d < dst; ++d) {
        auto s = static_cast<std::int32_t>(std::floor(d * scale));
        auto f = static_cast<float>((d + 1) - (s + 1) * inv_scale);
        f = f <= 0 ? 0.0F : f - std::floor(f);
        if (s + 1 >= src) {
            axis.limit = std::min(axis.limit, d);
            if (s >= src - 1) {
                f = 0.0F;
                s = src - 1;
            }
        }
        axis.index[static_cast<std::size_t>(d)] = s;
        // saturate_cast<short>(float) rounds to nearest even.
        axis.weight[static_cast<std::size_t>(d) * 2] =
            static_cast<std::int32_t>(std::nearbyint((1.0F - f) * static_cast<float>(kCoefScale)));
        axis.weight[static_cast<std::size_t>(d) * 2 + 1] =
            static_cast<std::int32_t>(std::nearbyint(f * static_cast<float>(kCoefScale)));
    }
    return axis;
}

} // namespace

std::vector<std::uint8_t> resize_area_up(const std::uint8_t* src, std::int32_t src_height,
                                         std::int32_t src_width, std::int32_t channels,
                                         std::int32_t height, std::int32_t width) {
    if (src_height <= 0 || src_width <= 0 || channels <= 0 || height < src_height ||
        width < src_width)
        throw std::invalid_argument("lerobot_groot resize_area_up only enlarges an image");
    const Axis xs = area_axis(src_width, width);
    const Axis ys = area_axis(src_height, height);
    const auto row_values = static_cast<std::size_t>(width) * channels;

    // Horizontal pass: integer sums scaled by 2^11; past the limit the border sample times one.
    std::vector<std::int32_t> rows(static_cast<std::size_t>(src_height) * row_values);
    for (std::int32_t y = 0; y < src_height; ++y) {
        const std::uint8_t* line = src + static_cast<std::size_t>(y) * src_width * channels;
        std::int32_t* out = rows.data() + static_cast<std::size_t>(y) * row_values;
        for (std::int32_t x = 0; x < width; ++x) {
            const auto s =
                static_cast<std::size_t>(xs.index[static_cast<std::size_t>(x)]) * channels;
            const std::int32_t a0 = xs.weight[static_cast<std::size_t>(x) * 2];
            const std::int32_t a1 = xs.weight[static_cast<std::size_t>(x) * 2 + 1];
            for (std::int32_t c = 0; c < channels; ++c) {
                out[static_cast<std::size_t>(x) * channels + c] =
                    x < xs.limit ? line[s + c] * a0 + line[s + channels + c] * a1
                                 : line[s + c] * kCoefScale;
            }
        }
    }

    // Vertical pass in OpenCV's 8-bit form: ((b0*(S0>>4))>>16) + ((b1*(S1>>4))>>16) + 2) >> 2.
    std::vector<std::uint8_t> result(static_cast<std::size_t>(height) * row_values);
    for (std::int32_t y = 0; y < height; ++y) {
        const std::int32_t s = ys.index[static_cast<std::size_t>(y)];
        const std::int32_t* r0 =
            rows.data() + static_cast<std::size_t>(std::min(s, src_height - 1)) * row_values;
        const std::int32_t* r1 =
            rows.data() + static_cast<std::size_t>(std::min(s + 1, src_height - 1)) * row_values;
        const std::int32_t b0 = ys.weight[static_cast<std::size_t>(y) * 2];
        const std::int32_t b1 = ys.weight[static_cast<std::size_t>(y) * 2 + 1];
        std::uint8_t* out = result.data() + static_cast<std::size_t>(y) * row_values;
        for (std::size_t i = 0; i < row_values; ++i) {
            const std::int32_t value =
                (((b0 * (r0[i] >> 4)) >> 16) + ((b1 * (r1[i] >> 4)) >> 16) + 2) >> 2;
            out[i] = static_cast<std::uint8_t>(std::clamp(value, 0, 255));
        }
    }
    return result;
}

void append_patches(const std::uint8_t* rgb, std::int32_t size, std::int32_t crop,
                    std::vector<float>& out) {
    if (size % (kPatch * kMerge) != 0 || crop <= 0 || crop > size)
        throw std::invalid_argument("lerobot_groot camera size or crop is not supported");
    std::vector<std::uint8_t> cropped;
    const std::uint8_t* image = rgb;
    std::vector<std::uint8_t> resized;
    if (crop < size) {
        const std::int32_t offset = (size - crop) / 2;
        cropped.resize(static_cast<std::size_t>(crop) * crop * 3);
        for (std::int32_t y = 0; y < crop; ++y)
            std::copy_n(rgb + (static_cast<std::size_t>(y + offset) * size + offset) * 3,
                        static_cast<std::size_t>(crop) * 3,
                        cropped.data() + static_cast<std::size_t>(y) * crop * 3);
        resized = resize_area_up(cropped.data(), crop, crop, 3, size, size);
        image = resized.data();
    }

    // Fused rescale + normalise of the processor: (x - 127.5) / 127.5 in float32. The temporal
    // patch repeats the single frame; values within a patch run channel, frame, row, column.
    const std::int32_t grid = size / kPatch;
    const std::int32_t blocks = grid / kMerge;
    for (std::int32_t block_row = 0; block_row < blocks; ++block_row) {
        for (std::int32_t block_col = 0; block_col < blocks; ++block_col) {
            for (std::int32_t inner_row = 0; inner_row < kMerge; ++inner_row) {
                for (std::int32_t inner_col = 0; inner_col < kMerge; ++inner_col) {
                    const std::int32_t top = (block_row * kMerge + inner_row) * kPatch;
                    const std::int32_t left = (block_col * kMerge + inner_col) * kPatch;
                    for (std::int32_t channel = 0; channel < 3; ++channel) {
                        for (std::int32_t frame = 0; frame < kTemporalPatch; ++frame) {
                            for (std::int32_t py = 0; py < kPatch; ++py) {
                                const std::uint8_t* line =
                                    image + (static_cast<std::size_t>(top + py) * size + left) * 3;
                                for (std::int32_t px = 0; px < kPatch; ++px)
                                    out.push_back(
                                        (static_cast<float>(
                                             line[static_cast<std::size_t>(px) * 3 + channel]) -
                                         127.5F) /
                                        127.5F);
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace trtmc::lerobot_groot
