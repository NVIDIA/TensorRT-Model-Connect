/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/vae_encoder_tiles.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

constexpr int32_t kMinimumOverlap = 64;
constexpr int32_t kInputAlignment = 32;
constexpr int32_t kInputChannels = 3;
constexpr int32_t kLatentTileSize = kMiniMaxH3VaeEncoderTileSize / kMiniMaxH3VaeSpatialCompression;

std::size_t checked_count(std::initializer_list<int32_t> dimensions, const char* label) {
    std::size_t result = 1;
    for (int32_t dimension : dimensions) {
        if (dimension <= 0 ||
            result > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension))
            throw std::invalid_argument(std::string(label) + " has invalid dimensions");
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

std::vector<int32_t> make_overlaps(int32_t length, int32_t tile_count) {
    std::vector<int32_t> result(static_cast<std::size_t>(tile_count - 1), kMinimumOverlap);
    int32_t remaining =
        kMiniMaxH3VaeEncoderTileSize * tile_count - kMinimumOverlap * (tile_count - 1) - length;
    if (remaining % kMiniMaxH3VaeSpatialCompression != 0)
        throw std::invalid_argument("MiniMax-H3 VAE tile length is not latent-aligned");
    for (int32_t index = 0; index < remaining / kMiniMaxH3VaeSpatialCompression; ++index)
        result[static_cast<std::size_t>(index % (tile_count - 1))] +=
            kMiniMaxH3VaeSpatialCompression;
    return result;
}

std::vector<MiniMaxH3VaeAxisTile> make_axis_plan(int32_t length) {
    if (length == kMiniMaxH3VaeEncoderTileSize) {
        return {{0, length, 0, length / kMiniMaxH3VaeSpatialCompression, 0, 0, 0,
                 length / kMiniMaxH3VaeSpatialCompression}};
    }
    int32_t tile_count = (length + kMiniMaxH3VaeEncoderTileSize - 1) / kMiniMaxH3VaeEncoderTileSize;
    while (kMiniMaxH3VaeEncoderTileSize * tile_count - kMinimumOverlap * (tile_count - 1) < length)
        ++tile_count;
    const auto overlaps = make_overlaps(length, tile_count);
    std::vector<MiniMaxH3VaeAxisTile> result;
    result.reserve(static_cast<std::size_t>(tile_count));
    int32_t start = 0;
    for (int32_t index = 0; index < tile_count; ++index) {
        const int32_t blend = index == 0 ? 0
                                         : overlaps[static_cast<std::size_t>(index - 1)] /
                                               kMiniMaxH3VaeSpatialCompression;
        const int32_t crop = index + 1 == tile_count ? 0
                                                     : overlaps[static_cast<std::size_t>(index)] /
                                                           kMiniMaxH3VaeSpatialCompression;
        result.push_back({start, kMiniMaxH3VaeEncoderTileSize,
                          start / kMiniMaxH3VaeSpatialCompression, kLatentTileSize, blend, crop,
                          start / kMiniMaxH3VaeSpatialCompression, kLatentTileSize - crop});
        if (index + 1 < tile_count)
            start += kMiniMaxH3VaeEncoderTileSize - overlaps[static_cast<std::size_t>(index)];
    }
    return result;
}

void validate_spatial_plan(const MiniMaxH3VaeSpatialTilePlan& plan) {
    if (plan.height < kMiniMaxH3VaeEncoderTileSize || plan.width < kMiniMaxH3VaeEncoderTileSize ||
        plan.height % kInputAlignment != 0 || plan.width % kInputAlignment != 0 ||
        plan.rows.empty() || plan.columns.empty())
        throw std::invalid_argument("MiniMax-H3 VAE spatial tile plan is invalid");
}

void validate_extraction_axis(const MiniMaxH3VaeAxisTile& axis, int32_t source_length) {
    if (axis.input_length != kMiniMaxH3VaeEncoderTileSize || axis.input_start < 0 ||
        axis.input_start + axis.input_length > source_length)
        throw std::invalid_argument("MiniMax-H3 VAE extraction axis is invalid");
}

void validate_extraction_temporal(const MiniMaxH3VaeTemporalChunk& temporal,
                                  int32_t source_frames) {
    if (temporal.valid_input_frames <= 0 || temporal.engine_num_frames <= 0 ||
        temporal.valid_input_frames + temporal.repeated_tail_frames != temporal.engine_num_frames)
        throw std::invalid_argument("MiniMax-H3 VAE extraction temporal chunk is invalid");
    if (temporal.input_start < 0 ||
        temporal.input_start + temporal.valid_input_frames > source_frames)
        throw std::invalid_argument("MiniMax-H3 VAE extraction temporal range is invalid");
    if (temporal.repeated_tail_frames > 0 &&
        (temporal.repeat_source_frame < 0 || temporal.repeat_source_frame >= source_frames))
        throw std::invalid_argument("MiniMax-H3 VAE extraction repeat frame is invalid");
}

std::size_t tile_offset(int32_t channel, int32_t frame, int32_t y, int32_t x, int32_t frames) {
    return (((static_cast<std::size_t>(channel) * frames + frame) * kLatentTileSize + y) *
            kLatentTileSize) +
           x;
}

void blend_vertical(std::vector<float>& current, const std::vector<float>& previous, int32_t frames,
                    int32_t extent) {
    for (int32_t channel = 0; channel < kMiniMaxH3VaeMomentChannels; ++channel) {
        for (int32_t frame = 0; frame < frames; ++frame) {
            for (int32_t y = 0; y < extent; ++y) {
                const float current_weight = static_cast<float>(y) / extent;
                for (int32_t x = 0; x < kLatentTileSize; ++x) {
                    const auto target = tile_offset(channel, frame, y, x, frames);
                    const auto source =
                        tile_offset(channel, frame, kLatentTileSize - extent + y, x, frames);
                    current[target] = previous[source] * (1.0F - current_weight) +
                                      current[target] * current_weight;
                }
            }
        }
    }
}

void blend_horizontal(std::vector<float>& current, const std::vector<float>& previous,
                      int32_t frames, int32_t extent) {
    for (int32_t channel = 0; channel < kMiniMaxH3VaeMomentChannels; ++channel) {
        for (int32_t frame = 0; frame < frames; ++frame) {
            for (int32_t y = 0; y < kLatentTileSize; ++y) {
                for (int32_t x = 0; x < extent; ++x) {
                    const float current_weight = static_cast<float>(x) / extent;
                    const auto target = tile_offset(channel, frame, y, x, frames);
                    const auto source =
                        tile_offset(channel, frame, y, kLatentTileSize - extent + x, frames);
                    current[target] = previous[source] * (1.0F - current_weight) +
                                      current[target] * current_weight;
                }
            }
        }
    }
}

void copy_stitched_tile(const std::vector<float>& tile, std::vector<float>& output,
                        const MiniMaxH3VaeAxisTile& row, const MiniMaxH3VaeAxisTile& column,
                        int32_t frames, int32_t output_height, int32_t output_width) {
    for (int32_t channel = 0; channel < kMiniMaxH3VaeMomentChannels; ++channel) {
        for (int32_t frame = 0; frame < frames; ++frame) {
            for (int32_t y = 0; y < row.stitch_length; ++y) {
                for (int32_t x = 0; x < column.stitch_length; ++x) {
                    const auto source = tile_offset(channel, frame, y, x, frames);
                    const auto target =
                        (((static_cast<std::size_t>(channel) * frames + frame) * output_height +
                          row.stitch_start + y) *
                             output_width +
                         column.stitch_start + x);
                    output[target] = tile[source];
                }
            }
        }
    }
}

} // namespace

MiniMaxH3VaeSpatialTilePlan make_minimax_h3_vae_spatial_tile_plan(int32_t height, int32_t width) {
    if (height < kMiniMaxH3VaeEncoderTileSize || width < kMiniMaxH3VaeEncoderTileSize ||
        height % kInputAlignment != 0 || width % kInputAlignment != 0)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA VAE canvas must be at least 256 and 32-aligned");
    return {height, width, make_axis_plan(height), make_axis_plan(width)};
}

MiniMaxH3VaeTemporalChunkPlan make_minimax_h3_vae_temporal_chunk_plan(int32_t num_frames) {
    if (num_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 VAE temporal plan requires frames");
    if (num_frames == 1)
        return {1, {{0, 1, 0, -1, 1, 0, 1}}, 1, 0, 1};
    const int32_t chunk_count =
        (num_frames + kMiniMaxH3VaeTemporalChunkFrames - 1) / kMiniMaxH3VaeTemporalChunkFrames;
    MiniMaxH3VaeTemporalChunkPlan result;
    result.num_frames = num_frames;
    result.raw_moment_frames = chunk_count * kMiniMaxH3VaeTemporalRawMoments;
    result.token_drop = kMiniMaxH3VaeTemporalTokenDrop;
    result.output_moment_frames = result.raw_moment_frames - result.token_drop;
    for (int32_t index = 0; index < chunk_count; ++index) {
        const int32_t start = index * kMiniMaxH3VaeTemporalChunkFrames;
        const int32_t valid = std::min(kMiniMaxH3VaeTemporalChunkFrames, num_frames - start);
        const int32_t repeated = kMiniMaxH3VaeTemporalChunkFrames - valid;
        result.chunks.push_back({start, valid, repeated, repeated == 0 ? -1 : num_frames - 1,
                                 kMiniMaxH3VaeTemporalChunkFrames,
                                 index * kMiniMaxH3VaeTemporalRawMoments,
                                 kMiniMaxH3VaeTemporalRawMoments});
    }
    return result;
}

std::vector<float> minimax_h3_extract_vae_encoder_tile(const std::vector<float>& normalized_rgb,
                                                       int32_t source_frames, int32_t source_height,
                                                       int32_t source_width,
                                                       const MiniMaxH3VaeTemporalChunk& temporal,
                                                       const MiniMaxH3VaeAxisTile& row,
                                                       const MiniMaxH3VaeAxisTile& column) {
    const auto expected =
        checked_count({kInputChannels, source_frames, source_height, source_width},
                      "MiniMax-H3 normalized reference video");
    if (normalized_rgb.size() != expected)
        throw std::invalid_argument("MiniMax-H3 VAE tile extraction contract is invalid");
    validate_extraction_axis(row, source_height);
    validate_extraction_axis(column, source_width);
    validate_extraction_temporal(temporal, source_frames);
    std::vector<float> result(
        checked_count({kInputChannels, temporal.engine_num_frames, kMiniMaxH3VaeEncoderTileSize,
                       kMiniMaxH3VaeEncoderTileSize},
                      "MiniMax-H3 VAE encoder tile"));
    for (int32_t channel = 0; channel < kInputChannels; ++channel) {
        for (int32_t frame = 0; frame < temporal.engine_num_frames; ++frame) {
            const int32_t source_frame = frame < temporal.valid_input_frames
                                             ? temporal.input_start + frame
                                             : temporal.repeat_source_frame;
            for (int32_t y = 0; y < kMiniMaxH3VaeEncoderTileSize; ++y) {
                const auto source =
                    (((static_cast<std::size_t>(channel) * source_frames + source_frame) *
                          source_height +
                      row.input_start + y) *
                         source_width +
                     column.input_start);
                const auto target =
                    (((static_cast<std::size_t>(channel) * temporal.engine_num_frames + frame) *
                          kMiniMaxH3VaeEncoderTileSize +
                      y) *
                     kMiniMaxH3VaeEncoderTileSize);
                std::copy_n(normalized_rgb.begin() + static_cast<std::ptrdiff_t>(source),
                            kMiniMaxH3VaeEncoderTileSize,
                            result.begin() + static_cast<std::ptrdiff_t>(target));
            }
        }
    }
    return result;
}

std::vector<float> minimax_h3_stitch_vae_encoder_tiles(const std::vector<std::vector<float>>& tiles,
                                                       const MiniMaxH3VaeSpatialTilePlan& plan,
                                                       int32_t raw_moment_frames) {
    validate_spatial_plan(plan);
    const std::size_t expected_tiles = plan.rows.size() * plan.columns.size();
    const auto tile_elements = checked_count(
        {kMiniMaxH3VaeMomentChannels, raw_moment_frames, kLatentTileSize, kLatentTileSize},
        "MiniMax-H3 raw VAE tile output");
    if (tiles.size() != expected_tiles ||
        !std::all_of(tiles.begin(), tiles.end(),
                     [tile_elements](const auto& tile) { return tile.size() == tile_elements; }))
        throw std::invalid_argument("MiniMax-H3 raw VAE tile outputs are incomplete");
    const int32_t output_height = plan.height / kMiniMaxH3VaeSpatialCompression;
    const int32_t output_width = plan.width / kMiniMaxH3VaeSpatialCompression;
    std::vector<float> output(
        checked_count({kMiniMaxH3VaeMomentChannels, raw_moment_frames, output_height, output_width},
                      "MiniMax-H3 stitched VAE moments"));
    for (std::size_t row = 0; row < plan.rows.size(); ++row) {
        for (std::size_t column = 0; column < plan.columns.size(); ++column) {
            const std::size_t tile_index = row * plan.columns.size() + column;
            auto current = tiles[tile_index];
            if (row > 0)
                blend_vertical(current, tiles[tile_index - plan.columns.size()], raw_moment_frames,
                               plan.rows[row].latent_blend_before);
            if (column > 0)
                blend_horizontal(current, tiles[tile_index - 1], raw_moment_frames,
                                 plan.columns[column].latent_blend_before);
            copy_stitched_tile(current, output, plan.rows[row], plan.columns[column],
                               raw_moment_frames, output_height, output_width);
        }
    }
    return output;
}

std::vector<float>
minimax_h3_assemble_vae_temporal_moments(const std::vector<std::vector<float>>& chunks,
                                         int32_t latent_height, int32_t latent_width,
                                         const MiniMaxH3VaeTemporalChunkPlan& plan) {
    if (chunks.size() != plan.chunks.size() || latent_height <= 0 || latent_width <= 0 ||
        plan.output_moment_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 VAE temporal chunks are invalid");
    const std::size_t plane = static_cast<std::size_t>(latent_height) * latent_width;
    std::vector<float> raw(checked_count(
        {kMiniMaxH3VaeMomentChannels, plan.raw_moment_frames, latent_height, latent_width},
        "MiniMax-H3 raw temporal moments"));
    for (std::size_t chunk = 0; chunk < chunks.size(); ++chunk) {
        const auto& metadata = plan.chunks[chunk];
        const std::size_t expected = static_cast<std::size_t>(kMiniMaxH3VaeMomentChannels) *
                                     metadata.raw_moment_frames * plane;
        if (chunks[chunk].size() != expected)
            throw std::invalid_argument("MiniMax-H3 stitched temporal chunk has wrong size");
        for (int32_t channel = 0; channel < kMiniMaxH3VaeMomentChannels; ++channel) {
            const auto source = chunks[chunk].begin() + static_cast<std::ptrdiff_t>(channel) *
                                                            metadata.raw_moment_frames * plane;
            const auto target =
                raw.begin() + (static_cast<std::ptrdiff_t>(channel) * plan.raw_moment_frames +
                               metadata.raw_moment_start) *
                                  static_cast<std::ptrdiff_t>(plane);
            std::copy_n(source, static_cast<std::size_t>(metadata.raw_moment_frames) * plane,
                        target);
        }
    }
    std::vector<float> output(checked_count(
        {kMiniMaxH3VaeMomentChannels, plan.output_moment_frames, latent_height, latent_width},
        "MiniMax-H3 temporal VAE moments"));
    for (int32_t channel = 0; channel < kMiniMaxH3VaeMomentChannels; ++channel) {
        const auto source =
            raw.begin() + static_cast<std::ptrdiff_t>(channel) * plan.raw_moment_frames * plane;
        const auto target = output.begin() + static_cast<std::ptrdiff_t>(channel) *
                                                 plan.output_moment_frames * plane;
        std::copy_n(source, static_cast<std::size_t>(plan.output_moment_frames) * plane, target);
    }
    return output;
}

} // namespace trtmc
