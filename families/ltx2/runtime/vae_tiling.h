/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Host side of the LTX-2.5 tiled video VAE decode (header-only so the contract tests compile it
// without engines). The build writes the tile plan (families/ltx2/vae_tiling.py) into
// runtime.json; every tile is decoded by one tile-shaped plan, and the tiles are blended here with
// linear ramps normalized by the summed weights. Each output value accumulates its tiles in tile
// order, so the result is the same bit for bit whichever rank decoded a tile.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace trtmc::ltx2 {

struct VaeTile {
    std::array<int32_t, 3> latent_start{}; // latent frame, row, column
    std::array<int32_t, 3> pixel_start{};  // output frame, row, column
    // (left, right) ramp lengths in output frames / pixels for time, height, width.
    std::array<std::array<int32_t, 2>, 3> ramps{};
    int32_t rank{0};
};

struct VaeTilePlan {
    std::array<int32_t, 3> tile_latent{}; // latent frames, rows, columns of every tile
    std::array<int32_t, 3> tile_pixels{}; // decoded frames, rows, columns of every tile
    std::vector<VaeTile> tiles;

    bool enabled() const { return !tiles.empty(); }
    std::size_t tile_values() const {
        return static_cast<std::size_t>(tile_pixels[0]) * tile_pixels[1] * tile_pixels[2] * 3U;
    }
};

// Ramp weights of one tile axis in fp32 (vae_tiling.py axis_weights). Spatial ramps fade in as
// k / (r + 1), k = 1..r; temporal left ramps fade in from 0 as k / r, k = 0..r-1; right ramps fade
// out as 1 - k / (r + 1), k = 1..r.
inline std::vector<float> vae_axis_weights(int32_t length, int32_t left, int32_t right,
                                           bool temporal) {
    if (length <= 0 || left < 0 || right < 0 || left > length || right > length)
        throw std::runtime_error("LTX-2.5 VAE tile ramp does not fit its tile");
    std::vector<float> w(static_cast<std::size_t>(length), 1.0F);
    for (int32_t k = 0; k < left; ++k) {
        w[static_cast<std::size_t>(k)] =
            temporal ? static_cast<float>(k) / static_cast<float>(left)
                     : static_cast<float>(k + 1) / static_cast<float>(left + 1);
    }
    for (int32_t k = 1; k <= right; ++k) {
        w[static_cast<std::size_t>(length - right + k - 1)] =
            1.0F - static_cast<float>(k) / static_cast<float>(right + 1);
    }
    return w;
}

// Checks that every tile lies inside the latent grid and the decoded video.
inline void vae_validate_plan(const VaeTilePlan& plan, const std::array<int32_t, 3>& latent,
                              const std::array<int32_t, 3>& video, int32_t world_size) {
    if (plan.tiles.empty())
        throw std::runtime_error("LTX-2.5 VAE tile plan has no tiles");
    for (const auto& tile : plan.tiles) {
        if (tile.rank < 0 || tile.rank >= world_size)
            throw std::runtime_error("LTX-2.5 VAE tile is assigned to a rank outside the world");
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (tile.latent_start[axis] < 0 ||
                tile.latent_start[axis] + plan.tile_latent[axis] > latent[axis] ||
                tile.pixel_start[axis] < 0 ||
                tile.pixel_start[axis] + plan.tile_pixels[axis] > video[axis])
                throw std::runtime_error("LTX-2.5 VAE tile lies outside the video");
            (void)vae_axis_weights(plan.tile_pixels[axis], tile.ramps[axis][0], tile.ramps[axis][1],
                                   axis == 0);
        }
    }
}

// Copies one tile's packed latent tokens [tf * th * tw, C] out of the packed video
// [F * H * W, C] (token order frame, row, column).
inline void vae_gather_tile_latents(const std::vector<float>& packed,
                                    const std::array<int32_t, 3>& latent, int32_t channels,
                                    const VaeTilePlan& plan, const VaeTile& tile,
                                    std::vector<float>& out) {
    const auto [tf, th, tw] = plan.tile_latent;
    const auto row = static_cast<std::size_t>(tw) * static_cast<std::size_t>(channels);
    out.resize(static_cast<std::size_t>(tf) * th * row);
    if (packed.size() != static_cast<std::size_t>(latent[0]) * latent[1] * latent[2] * channels)
        throw std::runtime_error("LTX-2.5 VAE tile gather: packed latents have the wrong size");
    std::size_t dst = 0;
    for (int32_t f = 0; f < tf; ++f) {
        for (int32_t h = 0; h < th; ++h) {
            const auto token = (static_cast<std::size_t>(tile.latent_start[0] + f) * latent[1] +
                                static_cast<std::size_t>(tile.latent_start[1] + h)) *
                                   latent[2] +
                               static_cast<std::size_t>(tile.latent_start[2]);
            std::memcpy(out.data() + dst, packed.data() + token * channels, row * sizeof(float));
            dst += row;
        }
    }
}

inline float vae_half_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000U) << 16U;
    const uint32_t exp = (h >> 10U) & 0x1FU;
    uint32_t mant = h & 0x3FFU;
    uint32_t bits = sign;
    if (exp == 31U) {
        bits |= 0x7F800000U | (mant << 13U);
    } else if (exp != 0U) {
        bits |= ((exp - 15U + 127U) << 23U) | (mant << 13U);
    } else if (mant != 0U) {
        int32_t e = -1;
        do {
            ++e;
            mant <<= 1U;
        } while ((mant & 0x400U) == 0U);
        bits |= (static_cast<uint32_t>(127 - 15 - e) << 23U) | ((mant & 0x3FFU) << 13U);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

namespace detail {

inline const std::vector<float>& half_table() {
    static const std::vector<float> table = [] {
        std::vector<float> values(65536);
        for (uint32_t i = 0; i < 65536U; ++i)
            values[i] = vae_half_to_float(static_cast<uint16_t>(i));
        return values;
    }();
    return table;
}

struct AxisWeights {
    std::vector<float> t, y, x;
};

// Accumulates one tile into the frame scratch and returns false when the tile misses the frame.
inline bool accumulate_tile(const VaeTilePlan& plan, const VaeTile& tile, const AxisWeights& w,
                            const uint16_t* values, int32_t frame, int32_t width,
                            std::vector<float>& num, std::vector<float>& den) {
    const auto [tt, th, tw] = plan.tile_pixels;
    const int32_t local_t = frame - tile.pixel_start[0];
    if (local_t < 0 || local_t >= tt)
        return false;
    const float wt = w.t[static_cast<std::size_t>(local_t)];
    if (wt == 0.0F)
        return true;
    const auto& lut = half_table();
    const uint16_t* src = values + static_cast<std::size_t>(local_t) * th * tw * 3U;
    for (int32_t y = 0; y < th; ++y) {
        const float wty = wt * w.y[static_cast<std::size_t>(y)];
        const auto out_row = static_cast<std::size_t>(tile.pixel_start[1] + y) * width +
                             static_cast<std::size_t>(tile.pixel_start[2]);
        float* n = num.data() + out_row * 3U;
        float* d = den.data() + out_row;
        for (int32_t x = 0; x < tw; ++x) {
            const float wxy = wty * w.x[static_cast<std::size_t>(x)];
            n[3 * x + 0] += wxy * lut[src[0]];
            n[3 * x + 1] += wxy * lut[src[1]];
            n[3 * x + 2] += wxy * lut[src[2]];
            d[x] += wxy;
            src += 3;
        }
    }
    return true;
}

} // namespace detail

// Blends decoded tiles (tiles[k]: fp16 [T, H, W, 3] of plan.tiles[k]) into clamped fp32
// [frames, height, width, 3]. Output frames are independent, so threads split the frames; every
// value accumulates its tiles in tile order whatever the thread count.
inline void vae_blend_tiles(const VaeTilePlan& plan, const std::vector<const uint16_t*>& tiles,
                            int32_t frames, int32_t height, int32_t width, std::vector<float>& out,
                            unsigned threads = 0) {
    if (tiles.size() != plan.tiles.size())
        throw std::runtime_error("LTX-2.5 VAE blend: one decoded buffer per tile is required");
    std::vector<detail::AxisWeights> weights(plan.tiles.size());
    for (std::size_t k = 0; k < plan.tiles.size(); ++k) {
        const auto& r = plan.tiles[k].ramps;
        weights[k].t = vae_axis_weights(plan.tile_pixels[0], r[0][0], r[0][1], true);
        weights[k].y = vae_axis_weights(plan.tile_pixels[1], r[1][0], r[1][1], false);
        weights[k].x = vae_axis_weights(plan.tile_pixels[2], r[2][0], r[2][1], false);
    }
    const auto plane = static_cast<std::size_t>(height) * width;
    out.resize(static_cast<std::size_t>(frames) * plane * 3U);
    if (threads == 0)
        threads = std::max(1U, std::min(32U, std::thread::hardware_concurrency()));
    threads = std::min<unsigned>(threads, static_cast<unsigned>(std::max(frames, 1)));
    auto work = [&](int32_t first, int32_t last) {
        std::vector<float> num(plane * 3U);
        std::vector<float> den(plane);
        for (int32_t f = first; f < last; ++f) {
            std::fill(num.begin(), num.end(), 0.0F);
            std::fill(den.begin(), den.end(), 0.0F);
            for (std::size_t k = 0; k < plan.tiles.size(); ++k)
                detail::accumulate_tile(plan, plan.tiles[k], weights[k], tiles[k], f, width, num,
                                        den);
            float* dst = out.data() + static_cast<std::size_t>(f) * plane * 3U;
            for (std::size_t p = 0; p < plane; ++p) {
                for (std::size_t c = 0; c < 3U; ++c) {
                    const float v = num[p * 3U + c] / den[p];
                    dst[p * 3U + c] = std::min(1.0F, std::max(0.0F, v));
                }
            }
        }
    };
    std::vector<std::thread> pool;
    const int32_t chunk =
        (frames + static_cast<int32_t>(threads) - 1) / static_cast<int32_t>(threads);
    for (unsigned i = 1; i < threads; ++i) {
        const int32_t first = static_cast<int32_t>(i) * chunk;
        if (first >= frames)
            break;
        pool.emplace_back(work, first, std::min(frames, first + chunk));
    }
    work(0, std::min(frames, chunk));
    for (auto& thread : pool)
        thread.join();
}

} // namespace trtmc::ltx2
