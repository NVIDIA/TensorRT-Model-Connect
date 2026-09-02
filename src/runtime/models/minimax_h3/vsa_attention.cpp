/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/vsa_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtmc::minimax_h3::vsa {
namespace {

int32_t ceil_div(int32_t value, int32_t divisor) {
    return (value + divisor - 1) / divisor;
}

void append_prefix_segment(TileLayout& layout, int32_t begin, int32_t count) {
    int32_t consumed = 0;
    while (consumed < count) {
        const int32_t valid = std::min(kTileTokens, count - consumed);
        layout.valid_sizes.push_back(valid);
        for (int32_t index = 0; index < valid; ++index)
            layout.tiled_to_packed.push_back(begin + consumed + index);
        layout.tiled_to_packed.insert(layout.tiled_to_packed.end(), kTileTokens - valid, -1);
        consumed += valid;
    }
}

} // namespace

Geometry make_geometry(int32_t num_frames, int32_t text_tokens) {
    if (text_tokens < 1 || text_tokens > 1024)
        throw std::invalid_argument("FastH3 VSA text token count must be in [1, 1024]");

    Geometry result;
    result.num_frames = num_frames;
    result.text_tokens = text_tokens;
    if (num_frames == 124) {
        result.audio_tokens = 414;
        result.video_latent_frames = 37;
    } else if (num_frames == 345) {
        result.audio_tokens = 1150;
        result.video_latent_frames = 102;
    } else {
        throw std::invalid_argument("FastH3 native VSA initially supports only 124 or 345 frames");
    }

    result.video_tokens = result.video_latent_frames * kVideoHeight * kVideoWidth;
    result.prefix_tiles =
        ceil_div(result.text_tokens, kTileTokens) + ceil_div(result.audio_tokens, kTileTokens);
    result.video_tiles = ceil_div(result.video_latent_frames, kVideoTileTime) *
                         ceil_div(kVideoHeight, kVideoTileHeight) *
                         ceil_div(kVideoWidth, kVideoTileWidth);
    result.total_tiles = result.prefix_tiles + result.video_tiles;
    result.top_video_tiles = std::max(1, ceil_div(result.video_tiles, 10));
    result.logical_rows = result.text_tokens + result.audio_tokens + result.video_tokens;
    result.padded_rows = result.total_tiles * kTileTokens;
    if (result.video_tiles > kMaxVideoTiles || result.top_video_tiles > kMaxTopVideoTiles)
        throw std::logic_error("FastH3 VSA geometry exceeds the compiled selector bucket");
    return result;
}

TileLayout make_tile_layout(int32_t num_frames, int32_t text_tokens) {
    TileLayout result;
    result.geometry = make_geometry(num_frames, text_tokens);
    result.tiled_to_packed.reserve(static_cast<std::size_t>(result.geometry.padded_rows));
    result.valid_sizes.reserve(static_cast<std::size_t>(result.geometry.total_tiles));

    append_prefix_segment(result, 0, result.geometry.text_tokens);
    append_prefix_segment(result, result.geometry.text_tokens, result.geometry.audio_tokens);

    const int32_t video_begin = result.geometry.text_tokens + result.geometry.audio_tokens;
    for (int32_t tile_t = 0; tile_t < result.geometry.video_latent_frames;
         tile_t += kVideoTileTime) {
        const int32_t end_t =
            std::min(tile_t + kVideoTileTime, result.geometry.video_latent_frames);
        for (int32_t tile_h = 0; tile_h < kVideoHeight; tile_h += kVideoTileHeight) {
            const int32_t end_h = std::min(tile_h + kVideoTileHeight, kVideoHeight);
            for (int32_t tile_w = 0; tile_w < kVideoWidth; tile_w += kVideoTileWidth) {
                const int32_t end_w = std::min(tile_w + kVideoTileWidth, kVideoWidth);
                const int32_t valid = (end_t - tile_t) * (end_h - tile_h) * (end_w - tile_w);
                result.valid_sizes.push_back(valid);
                for (int32_t time = tile_t; time < end_t; ++time) {
                    for (int32_t row = tile_h; row < end_h; ++row) {
                        for (int32_t column = tile_w; column < end_w; ++column) {
                            const int32_t raster =
                                (time * kVideoHeight + row) * kVideoWidth + column;
                            result.tiled_to_packed.push_back(video_begin + raster);
                        }
                    }
                }
                result.tiled_to_packed.insert(result.tiled_to_packed.end(), kTileTokens - valid,
                                              -1);
            }
        }
    }

    if (result.valid_sizes.size() != static_cast<std::size_t>(result.geometry.total_tiles) ||
        result.tiled_to_packed.size() != static_cast<std::size_t>(result.geometry.padded_rows)) {
        throw std::logic_error("FastH3 VSA tile builder produced an inconsistent layout");
    }
    return result;
}

std::vector<int32_t> select_video_topk_reference(const float* scores, int32_t heads,
                                                 int32_t total_tiles, int32_t prefix_tiles,
                                                 int32_t video_tiles, int32_t top_video_tiles) {
    if (scores == nullptr || heads <= 0 || total_tiles <= 0 || prefix_tiles < 0 ||
        video_tiles <= 0 || prefix_tiles + video_tiles != total_tiles || top_video_tiles <= 0 ||
        top_video_tiles > video_tiles) {
        throw std::invalid_argument("FastH3 VSA CPU selector received invalid geometry");
    }
    std::vector<int32_t> output(static_cast<std::size_t>(heads) * total_tiles * top_video_tiles,
                                -1);
    std::vector<std::pair<float, int32_t>> candidates;
    candidates.reserve(static_cast<std::size_t>(video_tiles));
    for (int32_t head = 0; head < heads; ++head) {
        for (int32_t query = 0; query < total_tiles; ++query) {
            candidates.clear();
            const std::size_t score_row =
                (static_cast<std::size_t>(head) * total_tiles + query) * total_tiles;
            for (int32_t video = 0; video < video_tiles; ++video) {
                const int32_t key = prefix_tiles + video;
                candidates.emplace_back(scores[score_row + key], key);
            }
            std::partial_sort(candidates.begin(), candidates.begin() + top_video_tiles,
                              candidates.end(), [](const auto& left, const auto& right) {
                                  if (left.first != right.first)
                                      return left.first > right.first;
                                  return left.second < right.second;
                              });
            auto output_row =
                output.begin() +
                (static_cast<std::size_t>(head) * total_tiles + query) * top_video_tiles;
            for (int32_t rank = 0; rank < top_video_tiles; ++rank)
                output_row[rank] = candidates[rank].second;
            std::sort(output_row, output_row + top_video_tiles);
        }
    }
    return output;
}

std::vector<int32_t> attended_key_tiles_reference(const int32_t* selected_video_tiles,
                                                  int32_t query_tile, int32_t prefix_tiles,
                                                  int32_t video_tiles, int32_t top_video_tiles) {
    const int32_t total_tiles = prefix_tiles + video_tiles;
    if (query_tile < 0 || query_tile >= total_tiles || prefix_tiles < 0 || video_tiles <= 0 ||
        top_video_tiles <= 0 || top_video_tiles > video_tiles || selected_video_tiles == nullptr) {
        throw std::invalid_argument("FastH3 VSA key-list reference received invalid geometry");
    }
    std::vector<int32_t> result;
    if (query_tile < prefix_tiles) {
        result.resize(static_cast<std::size_t>(total_tiles));
        std::iota(result.begin(), result.end(), 0);
        return result;
    }
    result.resize(static_cast<std::size_t>(prefix_tiles));
    std::iota(result.begin(), result.end(), 0);
    result.insert(result.end(), selected_video_tiles, selected_video_tiles + top_video_tiles);
    return result;
}

} // namespace trtmc::minimax_h3::vsa
