/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

inline constexpr int32_t kMiniMaxH3VaeEncoderTileSize = 256;
inline constexpr int32_t kMiniMaxH3VaeSpatialCompression = 16;
inline constexpr int32_t kMiniMaxH3VaeTemporalChunkFrames = 17;
inline constexpr int32_t kMiniMaxH3VaeTemporalRawMoments = 5;
inline constexpr int32_t kMiniMaxH3VaeTemporalTokenDrop = 3;
inline constexpr int32_t kMiniMaxH3VaeMomentChannels = 48;

struct MiniMaxH3VaeAxisTile {
    int32_t input_start{0};
    int32_t input_length{0};
    int32_t latent_start{0};
    int32_t latent_length{0};
    int32_t latent_blend_before{0};
    int32_t latent_crop_after{0};
    int32_t stitch_start{0};
    int32_t stitch_length{0};
};

struct MiniMaxH3VaeSpatialTilePlan {
    int32_t height{0};
    int32_t width{0};
    std::vector<MiniMaxH3VaeAxisTile> rows;
    std::vector<MiniMaxH3VaeAxisTile> columns;
};

struct MiniMaxH3VaeTemporalChunk {
    int32_t input_start{0};
    int32_t valid_input_frames{0};
    int32_t repeated_tail_frames{0};
    int32_t repeat_source_frame{-1};
    int32_t engine_num_frames{0};
    int32_t raw_moment_start{0};
    int32_t raw_moment_frames{0};
};

struct MiniMaxH3VaeTemporalChunkPlan {
    int32_t num_frames{0};
    std::vector<MiniMaxH3VaeTemporalChunk> chunks;
    int32_t raw_moment_frames{0};
    int32_t token_drop{0};
    int32_t output_moment_frames{0};
};

MiniMaxH3VaeSpatialTilePlan make_minimax_h3_vae_spatial_tile_plan(int32_t height, int32_t width);
MiniMaxH3VaeTemporalChunkPlan make_minimax_h3_vae_temporal_chunk_plan(int32_t num_frames);

// Extract normalized channel-major source pixels [3,T,H,W] into one static
// engine binding [1,3,engine_T,256,256], repeating the final source frame for
// a padded temporal tail.
std::vector<float> minimax_h3_extract_vae_encoder_tile(const std::vector<float>& normalized_rgb,
                                                       int32_t source_frames, int32_t source_height,
                                                       int32_t source_width,
                                                       const MiniMaxH3VaeTemporalChunk& temporal,
                                                       const MiniMaxH3VaeAxisTile& row,
                                                       const MiniMaxH3VaeAxisTile& column);

// Stitch row-major raw tile outputs. Every tile is channel-major
// [48,raw_T,16,16]. Blends read the unstitched upper/left neighbors, crop the
// current trailing overlap, concatenate width first, then height.
std::vector<float> minimax_h3_stitch_vae_encoder_tiles(const std::vector<std::vector<float>>& tiles,
                                                       const MiniMaxH3VaeSpatialTilePlan& plan,
                                                       int32_t raw_moment_frames);

// Concatenate channel-major stitched chunks along T and drop the last three
// temporal moments once from the combined video result.
std::vector<float>
minimax_h3_assemble_vae_temporal_moments(const std::vector<std::vector<float>>& chunks,
                                         int32_t latent_height, int32_t latent_width,
                                         const MiniMaxH3VaeTemporalChunkPlan& plan);

} // namespace trtmc
