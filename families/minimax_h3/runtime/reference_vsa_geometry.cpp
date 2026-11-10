/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_vsa_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::minimax_h3 {
namespace {
using Tile = std::vector<int32_t>;
using Region = std::vector<Tile>;

int64_t checked_rows(int32_t t, int32_t h, int32_t w) {
    int64_t rows = t;
    for (int32_t n : {h, w}) {
        if (rows > std::numeric_limits<int32_t>::max() / n)
            throw std::length_error("OmniRef VSA rows exceed the native index range");
        rows *= n;
    }
    return rows;
}

int64_t divide_up(int32_t value, int32_t divisor) {
    return (static_cast<int64_t>(value) + divisor - 1) / divisor;
}
} // namespace

ReferenceVsaLayout make_reference_vsa_layout(const std::vector<ReferenceVsaSegment>& segments,
                                             int32_t max_tiles) {
    if (segments.empty() || max_tiles < 2 || max_tiles > 8192 || max_tiles % 2)
        throw std::invalid_argument("OmniRef VSA requires segments and an even tile capacity");
    std::vector<Tile> dense;
    std::vector<Region> videos;
    int64_t cursor = 0;
    int64_t tile_count = 0;
    for (const auto& segment : segments) {
        if (const auto* rows = std::get_if<int32_t>(&segment)) {
            if (*rows < 0)
                throw std::invalid_argument("OmniRef VSA dense lengths must be nonnegative");
            if (cursor + *rows >= std::numeric_limits<int32_t>::max())
                throw std::length_error("OmniRef VSA rows exceed the native index range");
            tile_count += divide_up(*rows, 128);
            if (tile_count > max_tiles)
                throw std::length_error("OmniRef VSA geometry exceeds its loaded tile profile");
            for (int64_t offset = 0; offset < *rows; offset += 128) {
                Tile tile;
                for (int64_t row = offset; row < std::min<int64_t>(offset + 128, *rows); ++row)
                    tile.push_back(static_cast<int32_t>(cursor + row));
                dense.push_back(std::move(tile));
            }
            cursor += *rows;
            continue;
        }
        const auto grid = std::get<ReferenceVsaVideoGrid>(segment);
        if (grid.frames <= 0 || grid.height <= 0 || grid.width <= 0 || grid.height % 2 ||
            grid.width % 2)
            throw std::invalid_argument(
                "OmniRef VSA video requires positive (1,2,2)-aligned latents");
        const int32_t t = grid.frames, h = grid.height / 2, w = grid.width / 2;
        const int64_t rows = checked_rows(t, h, w);
        const int64_t tiles = divide_up(t, 4) * divide_up(h, 4) * divide_up(w, 8);
        if (tiles > 1024 || videos.size() >= 64 || tile_count + tiles > max_tiles ||
            cursor + rows >= std::numeric_limits<int32_t>::max())
            throw std::length_error("OmniRef VSA video exceeds its routing profile");
        Region region;
        for (int32_t a = 0; a < t; a += 4)
            for (int32_t b = 0; b < h; b += 4)
                for (int32_t c = 0; c < w; c += 8) {
                    Tile tile;
                    for (int32_t i = a; i < std::min(a + 4, t); ++i)
                        for (int32_t j = b; j < std::min(b + 4, h); ++j)
                            for (int32_t k = c; k < std::min(c + 8, w); ++k)
                                tile.push_back(static_cast<int32_t>(
                                    cursor + (static_cast<int64_t>(i) * h + j) * w + k));
                    region.push_back(std::move(tile));
                }
        videos.push_back(std::move(region));
        cursor += rows;
        tile_count += tiles;
    }
    if (videos.empty())
        throw std::invalid_argument("OmniRef VSA requires a generated video region");
    ReferenceVsaLayout result;
    result.source_rows = static_cast<int32_t>(cursor);
    result.prefix_tiles = static_cast<int32_t>(dense.size());
    auto tiles = std::move(dense);
    for (auto& region : videos) {
        const int32_t begin = static_cast<int32_t>(tiles.size());
        const int32_t count = static_cast<int32_t>(region.size());
        const int32_t keep =
            std::max(1, std::min(count, static_cast<int32_t>(std::ceil((1.0 - 0.9) * count))));
        result.regions.insert(result.regions.end(), {begin, begin + count, keep});
        for (auto& tile : region)
            tiles.push_back(std::move(tile));
    }
    result.logical_tiles = static_cast<int32_t>(tiles.size());
    if (tiles.size() % 2)
        tiles.emplace_back();
    if (tiles.size() > static_cast<std::size_t>(max_tiles))
        throw std::length_error("OmniRef VSA partner tile exceeds its loaded profile");
    result.untile_indices.resize(static_cast<std::size_t>(cursor));
    result.gather_indices.assign(tiles.size() * 128, result.source_rows);
    std::vector<bool> seen(static_cast<std::size_t>(cursor), false);
    std::size_t live = 0;
    for (std::size_t index = 0; index < tiles.size(); ++index) {
        result.block_sizes.push_back(static_cast<int32_t>(tiles[index].size()));
        for (std::size_t offset = 0; offset < tiles[index].size(); ++offset) {
            const auto row = tiles[index][offset];
            if (row < 0 || row >= result.source_rows || seen[static_cast<std::size_t>(row)])
                throw std::runtime_error("OmniRef VSA row mapping is not injective");
            const auto slot = static_cast<int32_t>(index * 128 + offset);
            result.gather_indices[static_cast<std::size_t>(slot)] = row;
            result.untile_indices[static_cast<std::size_t>(row)] = slot;
            seen[static_cast<std::size_t>(row)] = true;
            ++live;
        }
    }
    if (live != static_cast<std::size_t>(cursor))
        throw std::runtime_error("OmniRef VSA did not cover every packed row");
    return result;
}

ReferenceVsaLayout make_reference_vsa_layout(int32_t text_rows,
                                             const std::vector<ImageReferenceLatentShape>& images,
                                             ReferenceTargetLatentShape target,
                                             std::optional<ReferenceContinuationGuideShape> guide,
                                             int32_t max_tiles) {
    if (text_rows <= 0 || target.audio_latents <= 0 ||
        target.audio_latents > std::numeric_limits<int32_t>::max() / 2 ||
        (guide && (guide->audio_latents <= 0 ||
                   guide->audio_latents > std::numeric_limits<int32_t>::max() / 2)))
        throw std::invalid_argument("OmniRef VSA requires positive bounded text/audio segments");
    std::vector<ReferenceVsaSegment> segments{text_rows};
    for (const auto& image : images) {
        if (image.height <= 0 || image.width <= 0 || image.height % 2 || image.width % 2)
            throw std::invalid_argument("OmniRef VSA images require patch-aligned latent geometry");
        const int64_t rows = static_cast<int64_t>(image.height / 2) * (image.width / 2);
        if (rows >= std::numeric_limits<int32_t>::max())
            throw std::length_error("OmniRef VSA image exceeds the native row index range");
        segments.emplace_back(static_cast<int32_t>(rows));
    }
    if (guide) {
        segments.emplace_back(
            ReferenceVsaVideoGrid{guide->video_frames, target.height, target.width});
        segments.emplace_back(2 * guide->audio_latents);
    }
    segments.emplace_back(2 * target.audio_latents);
    segments.emplace_back(ReferenceVsaVideoGrid{target.frames, target.height, target.width});
    return make_reference_vsa_layout(segments, max_tiles);
}

} // namespace trtmc::minimax_h3
