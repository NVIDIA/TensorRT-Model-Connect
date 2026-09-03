/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <vector>

namespace trtmc::minimax_h3::vsa {

// FastH3's trained VSA tile geometry. The segmented runtime passes dynamic
// valid-size metadata directly to these CUDA kernels; its capacity covers every
// released 768p canvas from aspect ratio 1:4 through 4:1 after the official
// multiple-of-32 rounding.
inline constexpr int32_t kTileTokens = 64;
inline constexpr int32_t kHeads = 56;
inline constexpr int32_t kHeadDim = 128;
inline constexpr int32_t kVideoTileTime = 4;
inline constexpr int32_t kVideoTileHeight = 4;
inline constexpr int32_t kVideoTileWidth = 4;
inline constexpr int32_t kVideoHeight = 24;
inline constexpr int32_t kVideoWidth = 42;
// At 345 output frames, the worst rounded public canvas requires 2,080
// (4,4,4) video tiles. The qualified 768x1344 canvas remains 1,716 tiles.
inline constexpr int32_t kMaxVideoTiles = 2080;
inline constexpr int32_t kMaxTopVideoTiles = 208;

struct Geometry {
    int32_t num_frames{0};
    int32_t text_tokens{0};
    int32_t audio_tokens{0};
    int32_t video_latent_frames{0};
    int32_t video_tokens{0};
    int32_t prefix_tiles{0};
    int32_t video_tiles{0};
    int32_t total_tiles{0};
    int32_t top_video_tiles{0};
    int32_t logical_rows{0};
    int32_t padded_rows{0};
};

struct TileLayout {
    Geometry geometry;
    // For every tile-major padded row, the corresponding packed
    // [text|audio|video-raster] row, or -1 for padding.
    std::vector<int32_t> tiled_to_packed;
    // Number of valid rows in every 64-token tile.
    std::vector<int32_t> valid_sizes;
};

// Build the exact segment-pure prefix and (4,4,4) video tile layout used by
// FastH3. num_frames must be 124 or 345; text_tokens may be 1..1024.
Geometry make_geometry(int32_t num_frames, int32_t text_tokens);
TileLayout make_tile_layout(int32_t num_frames, int32_t text_tokens);

// CPU reference for the selector. scores is row-major [heads, tiles, tiles].
// The output is [heads, tiles, top_video_tiles], contains absolute video-tile
// indices, and is sorted by tile index to match map_to_index semantics.
std::vector<int32_t> select_video_topk_reference(const float* scores, int32_t heads,
                                                 int32_t total_tiles, int32_t prefix_tiles,
                                                 int32_t video_tiles, int32_t top_video_tiles);

// Materialize the key-tile list used by one query tile. Prefix queries are
// dense. Video queries attend every prefix tile plus their selected video
// tiles. This helper is intentionally CPU-only and used by strict references.
std::vector<int32_t> attended_key_tiles_reference(const int32_t* selected_video_tiles,
                                                  int32_t query_tile, int32_t prefix_tiles,
                                                  int32_t video_tiles, int32_t top_video_tiles);

// All device tensors below use batch=1 and a tile-major BHSD-equivalent
// layout [heads, tiles, 64, 128]. These launches are forward-only and enqueue
// on the caller's stream without synchronizing.
void tile_bhsd_async(const __nv_bfloat16* packed, const int32_t* tiled_to_packed,
                     __nv_bfloat16* tiled, int32_t heads, int32_t logical_rows, int32_t total_tiles,
                     cudaStream_t stream);

void untile_bhsd_async(const __nv_bfloat16* tiled, const int32_t* tiled_to_packed,
                       __nv_bfloat16* packed, int32_t heads, int32_t logical_rows,
                       int32_t total_tiles, cudaStream_t stream);

// FP32 mean pooling over valid rows: tiled BF16 -> [heads, tiles, 128].
void mean_pool_tiles_async(const __nv_bfloat16* tiled, const int32_t* valid_sizes, float* pooled,
                           int32_t heads, int32_t total_tiles, cudaStream_t stream);

// Concatenate segment-pure prefix and video valid-size arrays into the one
// tile array consumed by the pooled and sparse kernels.
void concatenate_valid_sizes_async(const int32_t* prefix_valid_sizes,
                                   const int32_t* video_valid_sizes, int32_t* valid_sizes,
                                   int32_t prefix_tiles, int32_t video_tiles, cudaStream_t stream);

// Compute scaled pooled QK scores [heads, tiles, tiles] with a CUDA kernel.
// This deliberately avoids a cuBLAS runtime/distribution dependency.
void pooled_qk_scores_async(const float* pooled_q, const float* pooled_k, float* scores,
                            int32_t heads, int32_t total_tiles, cudaStream_t stream);

// Select ceil(10% * video_tiles) video key tiles for every (head, query tile).
// The compact output is sorted by absolute tile index.
void select_video_topk_async(const float* scores, int32_t* selected_video_tiles, int32_t heads,
                             int32_t total_tiles, int32_t prefix_tiles, int32_t video_tiles,
                             int32_t top_video_tiles, cudaStream_t stream);

// Tensor-core 64x64 block-sparse attention. Prefix query tiles are dense;
// video query tiles use prefix + selected video keys. Invalid padded query
// rows are written as exact zero.
void block_sparse_attention_64_async(const __nv_bfloat16* query, const __nv_bfloat16* key,
                                     const __nv_bfloat16* value, const int32_t* valid_sizes,
                                     const int32_t* selected_video_tiles, __nv_bfloat16* output,
                                     int32_t heads, int32_t total_tiles, int32_t prefix_tiles,
                                     int32_t video_tiles, int32_t top_video_tiles,
                                     cudaStream_t stream);

struct Sm121AttentionWorkspace {
    int32_t* q2k_index{nullptr};
    std::size_t q2k_index_capacity{0};
    int32_t* q2k_count{nullptr};
    std::size_t q2k_count_capacity{0};
    float* lse{nullptr};
    std::size_t lse_capacity{0};
};

// Launch the embedded-PTX SM121 specialization. Each workspace must remain
// exclusively owned by this stream through completion; capacities are element
// counts, not bytes. Throws if the specialization is unavailable.
void block_sparse_attention_64_sm121_async(
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    const int32_t* valid_sizes, const int32_t* selected_video_tiles, __nv_bfloat16* output,
    int32_t heads, int32_t total_tiles, int32_t prefix_tiles, int32_t video_tiles,
    int32_t top_video_tiles, cudaStream_t stream, const Sm121AttentionWorkspace& workspace);

// Report whether the embedded SM121 online-attention specialization was built,
// applies to the current device, and loaded successfully.
enum class Sm121AttentionStatus {
    kNotBuilt,
    kUnsupportedDevice,
    kReady,
    kLoadFailed,
};

Sm121AttentionStatus block_sparse_attention_sm121_status();
bool block_sparse_attention_sm121_available();

// Dense pooled attention used by FastH3's learned compression branch:
// softmax(scores) @ pooled_v -> [heads, tiles, 128], in FP32.
void pooled_gate_attention_async(const float* scores, const float* pooled_v, float* compressed,
                                 int32_t heads, int32_t total_tiles, cudaStream_t stream);

// In-place-equivalent gate merge written to output:
// output = sparse + gate * compressed[tile]. Padded gate rows are expected to
// be zero, as produced by tile_bhsd_async.
void merge_gate_async(const __nv_bfloat16* sparse, const __nv_bfloat16* gate,
                      const float* compressed, __nv_bfloat16* output, int32_t heads,
                      int32_t total_tiles, cudaStream_t stream);

} // namespace trtmc::minimax_h3::vsa
