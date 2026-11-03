/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/video_decode_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::minimax_h3 {
namespace {
constexpr std::size_t kTile = 256, kTileFrames = 28, kChannels = 3, kOverlap = 5;
constexpr std::size_t kTileValues = kChannels * kTileFrames * kTile * kTile;
constexpr std::array<float, 3> kMean = {0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kStd = {0.229F, 0.224F, 0.225F};

std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("MiniMax-H3 decoded video size overflow");
    return a * b;
}

uint16_t blend(uint16_t a, uint16_t b, std::size_t index, std::size_t extent) {
    // torch.arange(dtype=half) / extent is rounded BEFORE 1 - weight.
    const float wb = video_half_to_float(
        video_float_to_half(static_cast<float>(index) / static_cast<float>(extent)));
    const float wa = video_half_to_float(video_float_to_half(1.0F - wb));
    const float lhs = video_half_to_float(video_float_to_half(video_half_to_float(a) * wa));
    const float rhs = video_half_to_float(video_float_to_half(video_half_to_float(b) * wb));
    return video_float_to_half(lhs + rhs);
}

std::size_t tile_offset(std::size_t channel, std::size_t frame, std::size_t y, std::size_t x) {
    return ((channel * kTileFrames + frame) * kTile + y) * kTile + x;
}
} // namespace

uint16_t video_float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const auto sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t mantissa = bits & 0x7fffffU;
    const uint32_t exponent = (bits >> 23) & 255U;
    if (exponent == 255)
        return sign | (mantissa ? 0x7e00U : 0x7c00U);
    int32_t adjusted = static_cast<int32_t>(exponent) - 127 + 15;
    if (adjusted >= 31)
        return sign | 0x7c00U;
    if (adjusted <= 0) {
        if (adjusted < -10)
            return sign;
        const uint32_t significand = mantissa | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - adjusted);
        const uint32_t remainder = significand & ((1U << shift) - 1U);
        uint32_t rounded = significand >> shift;
        const uint32_t midpoint = 1U << (shift - 1U);
        if (remainder > midpoint || (remainder == midpoint && (rounded & 1U)))
            ++rounded;
        return sign | static_cast<uint16_t>(rounded);
    }
    uint32_t rounded = (mantissa + 0xfffU + ((mantissa >> 13) & 1U)) >> 13;
    if (rounded == 0x400U) {
        rounded = 0;
        ++adjusted;
    }
    return sign | static_cast<uint16_t>((static_cast<uint32_t>(adjusted) << 10) | rounded);
}

float video_half_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t mantissa = value & 0x3ffU;
    const uint32_t exponent = (value >> 10) & 31U;
    uint32_t bits = sign;
    if (exponent == 31) {
        bits |= 0x7f800000U | (mantissa << 13);
    } else if (exponent) {
        bits |= ((exponent + 112U) << 23) | (mantissa << 13);
    } else if (mantissa) {
        int32_t adjusted = -14;
        while (!(mantissa & 0x400U)) {
            mantissa <<= 1;
            --adjusted;
        }
        bits |= (static_cast<uint32_t>(adjusted + 127) << 23) | ((mantissa & 0x3ffU) << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

VideoDecodeGeometry make_video_decode_geometry(int32_t latent_frames, int32_t latent_height,
                                               int32_t latent_width) {
    if (latent_frames < 3 || latent_height < 16 || latent_width < 16 || latent_height % 2 ||
        latent_width % 2 || latent_height > std::numeric_limits<int32_t>::max() / 16 ||
        latent_width > std::numeric_limits<int32_t>::max() / 16)
        throw std::invalid_argument(
            "MiniMax-H3 video decode requires valid original latent geometry");
    const int64_t tokens = static_cast<int64_t>(latent_frames) + 3;
    const int64_t padding = (5 - tokens % 5) % 5;
    const int64_t clips = (tokens + padding) / 5 - 1;
    int64_t trimmed = 0;
    for (int64_t k = 0; k < padding; ++k)
        trimmed += (static_cast<int64_t>(latent_frames) + k) % 5 == 0 ? 1 : 4;
    const int64_t frames = clips * 17 + 5 - trimmed;
    if (clips <= 0 || frames <= 0 || frames > std::numeric_limits<int32_t>::max())
        throw std::overflow_error("MiniMax-H3 decoded frame count overflow");
    const int32_t height = latent_height * 16, width = latent_width * 16;
    (void)product(product(product(static_cast<std::size_t>(frames), height), width),
                  kChannels * sizeof(float));
    return {latent_frames,
            latent_height,
            latent_width,
            static_cast<int32_t>(frames),
            height,
            width,
            static_cast<int32_t>(clips),
            static_cast<int32_t>(padding),
            make_visual_reference_axis(height),
            make_visual_reference_axis(width)};
}

struct VideoClipAssembler::State {
    VideoDecodeGeometry geometry;
    DecodedH3Video output;
    std::vector<std::vector<uint16_t>> above;
    std::vector<uint16_t> overlap;
    int32_t next_clip{0};
    std::size_t next_tile{0};
    bool poisoned{false}, finished{false};

    explicit State(VideoDecodeGeometry g)
        : geometry(std::move(g)), output{geometry.frames, geometry.height, geometry.width, {}},
          above(geometry.horizontal.starts.size()) {
        output.rgb.resize(
            product(product(product(geometry.frames, geometry.height), geometry.width), kChannels));
        overlap.resize(
            product(product(product(kChannels * kOverlap, geometry.height), geometry.width), 1));
    }

    void write_pixel(std::size_t channel, std::size_t frame, std::size_t y, std::size_t x,
                     uint16_t value) {
        if (frame >= static_cast<std::size_t>(geometry.frames))
            return; // Original trailing repeated-latent crop, not a preview crop.
        volatile float scaled = video_half_to_float(value) * kStd[channel];
        const float pixel = scaled + kMean[channel];
        if (!std::isfinite(pixel))
            throw std::runtime_error("MiniMax-H3 video blending produced a non-finite pixel");
        output.rgb[((frame * geometry.height + y) * geometry.width + x) * kChannels + channel] =
            std::clamp(pixel, 0.0F, 1.0F);
    }
};

VideoClipAssembler::VideoClipAssembler(VideoDecodeGeometry geometry) {
    const auto expected = make_video_decode_geometry(geometry.latent_frames, geometry.latent_height,
                                                     geometry.latent_width);
    if (expected.frames != geometry.frames || expected.height != geometry.height ||
        expected.width != geometry.width || expected.clips != geometry.clips ||
        expected.pad_tokens != geometry.pad_tokens ||
        expected.vertical.starts != geometry.vertical.starts ||
        expected.horizontal.starts != geometry.horizontal.starts ||
        expected.vertical.lengths != geometry.vertical.lengths ||
        expected.horizontal.lengths != geometry.horizontal.lengths ||
        expected.vertical.overlaps != geometry.vertical.overlaps ||
        expected.horizontal.overlaps != geometry.horizontal.overlaps)
        throw std::invalid_argument("MiniMax-H3 video decode geometry was modified");
    state_ = std::make_unique<State>(std::move(geometry));
}
VideoClipAssembler::~VideoClipAssembler() = default;

void VideoClipAssembler::append(int32_t clip, std::size_t tile, const float* decoded,
                                std::size_t count) {
    auto& s = *state_;
    const auto& g = s.geometry;
    if (s.finished || s.poisoned || clip != s.next_clip || tile != s.next_tile || clip >= g.clips ||
        tile >= g.tiles() || !decoded || count != kTileValues)
        throw std::invalid_argument(
            "MiniMax-H3 video tiles must arrive in exact clip/row-major order");
    s.poisoned = true;
    std::vector<uint16_t> raw(count);
    for (std::size_t i = 0; i < count; ++i) {
        raw[i] = video_float_to_half(decoded[i]);
        const float widened = video_half_to_float(raw[i]);
        if (!std::isfinite(decoded[i]) || std::memcmp(&widened, decoded + i, sizeof(float)) != 0)
            throw std::runtime_error(
                "MiniMax-H3 decoded tile must be a lossless FP32 widening of FP16");
    }
    const std::size_t row = tile / g.horizontal.starts.size(),
                      col = tile % g.horizontal.starts.size();
    const auto y_begin = static_cast<std::size_t>(g.vertical.starts[row]);
    const auto x_begin = static_cast<std::size_t>(g.horizontal.starts[col]);
    const auto y_kept = kTile - (row + 1 < g.vertical.starts.size() ? g.vertical.overlaps[row] : 0);
    const auto x_kept =
        kTile - (col + 1 < g.horizontal.starts.size() ? g.horizontal.overlaps[col] : 0);
    const auto y_overlap = row ? static_cast<std::size_t>(g.vertical.overlaps[row - 1]) : 0;
    const auto x_overlap = col ? static_cast<std::size_t>(g.horizontal.overlaps[col - 1]) : 0;
    for (std::size_t channel = 0; channel < kChannels; ++channel)
        for (std::size_t frame = 3; frame < kTileFrames; ++frame) {
            if (frame >= 20 && frame < 23)
                continue;
            for (std::size_t y = 0; y < y_kept; ++y)
                for (std::size_t x = 0; x < x_kept; ++x) {
                    uint16_t value = raw[tile_offset(channel, frame, y, x)];
                    if (row && y < y_overlap)
                        value = blend(
                            s.above[col][tile_offset(channel, frame, kTile - y_overlap + y, x)],
                            value, y, y_overlap);
                    if (col && x < x_overlap)
                        value = blend(
                            s.above[col - 1][tile_offset(channel, frame, y, kTile - x_overlap + x)],
                            value, x, x_overlap);
                    const auto pixel =
                        (static_cast<std::size_t>(channel) * kOverlap * g.height + y_begin + y) *
                            g.width +
                        x_begin + x;
                    if (frame < 20) {
                        const auto time = frame - 3;
                        if (clip && time < kOverlap)
                            value = blend(s.overlap[pixel + time * g.height * g.width], value, time,
                                          kOverlap);
                        s.write_pixel(channel, static_cast<std::size_t>(clip) * 17 + time,
                                      y_begin + y, x_begin + x, value);
                    } else {
                        s.overlap[pixel + (frame - 23) * g.height * g.width] = value;
                    }
                }
        }
    s.above[col] = std::move(raw); // ORIGINAL current tile, not its blended/cropped values.
    if (++s.next_tile == g.tiles()) {
        s.next_tile = 0;
        ++s.next_clip;
    }
    s.poisoned = false;
}

DecodedH3Video VideoClipAssembler::finish() {
    auto& s = *state_;
    const auto& g = s.geometry;
    if (s.finished || s.poisoned || s.next_clip != g.clips || s.next_tile)
        throw std::logic_error("MiniMax-H3 video assembly is incomplete or failed");
    for (std::size_t c = 0; c < kChannels; ++c)
        for (std::size_t t = 0; t < kOverlap; ++t)
            for (std::size_t y = 0; y < static_cast<std::size_t>(g.height); ++y)
                for (std::size_t x = 0; x < static_cast<std::size_t>(g.width); ++x)
                    s.write_pixel(c, static_cast<std::size_t>(g.clips) * 17 + t, y, x,
                                  s.overlap[((c * kOverlap + t) * g.height + y) * g.width + x]);
    s.finished = true;
    return std::move(s.output);
}

} // namespace trtmc::minimax_h3
