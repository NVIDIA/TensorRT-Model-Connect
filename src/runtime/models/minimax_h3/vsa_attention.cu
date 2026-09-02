/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/vsa_attention.h"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <limits>
#include <mma.h>
#include <stdexcept>
#include <string>

namespace trtmc::minimax_h3::vsa {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int32_t kWarpSize = 32;
constexpr int32_t kAttentionThreads = 256;
constexpr int32_t kSelectorThreads = 256;
// The public worst-aspect 15-second canvas has 2,080 video tiles. Nine
// blocked items per thread keep the one-block radix selector fail-closed above
// that profile while covering all released geometry (capacity 2,304).
constexpr int32_t kSelectorItems = 9;
constexpr int32_t kSelectorCapacity = kSelectorThreads * kSelectorItems;
constexpr float kAttentionScale = 0.08838834764831844055F; // 1 / sqrt(128)
constexpr float kLog2E = 1.4426950408889634074F;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_launch(const char* operation) {
    check_cuda(cudaPeekAtLastError(), operation);
}

void validate_tiled(const void* input, const void* output, int32_t heads, int32_t logical_rows,
                    int32_t total_tiles) {
    if (input == nullptr || output == nullptr || heads <= 0 || logical_rows <= 0 ||
        total_tiles <= 0) {
        throw std::invalid_argument("FastH3 VSA tiled launch received invalid arguments");
    }
}

void validate_sparse_geometry(int32_t heads, int32_t total_tiles, int32_t prefix_tiles,
                              int32_t video_tiles, int32_t top_video_tiles) {
    if (heads <= 0 || total_tiles <= 0 || prefix_tiles < 0 || video_tiles <= 0 ||
        prefix_tiles + video_tiles != total_tiles || video_tiles > kMaxVideoTiles ||
        top_video_tiles <= 0 || top_video_tiles > video_tiles ||
        top_video_tiles > kMaxTopVideoTiles) {
        throw std::invalid_argument("FastH3 VSA launch received unsupported geometry");
    }
}

__global__ void tile_bhsd_kernel(const __nv_bfloat16* packed, const int32_t* tiled_to_packed,
                                 __nv_bfloat16* tiled, int32_t logical_rows, int32_t padded_rows,
                                 int32_t heads) {
    const std::int64_t count = static_cast<std::int64_t>(heads) * padded_rows * kHeadDim;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int32_t dimension = static_cast<int32_t>(index % kHeadDim);
        const std::int64_t row_index = index / kHeadDim;
        const int32_t padded_row = static_cast<int32_t>(row_index % padded_rows);
        const int32_t head = static_cast<int32_t>(row_index / padded_rows);
        const int32_t packed_row = tiled_to_packed[padded_row];
        tiled[index] =
            packed_row >= 0
                ? packed[(static_cast<std::int64_t>(head) * logical_rows + packed_row) * kHeadDim +
                         dimension]
                : __float2bfloat16_rn(0.0F);
    }
}

__global__ void untile_bhsd_kernel(const __nv_bfloat16* tiled, const int32_t* tiled_to_packed,
                                   __nv_bfloat16* packed, int32_t logical_rows, int32_t padded_rows,
                                   int32_t heads) {
    const std::int64_t count = static_cast<std::int64_t>(heads) * padded_rows * kHeadDim;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int32_t dimension = static_cast<int32_t>(index % kHeadDim);
        const std::int64_t row_index = index / kHeadDim;
        const int32_t padded_row = static_cast<int32_t>(row_index % padded_rows);
        const int32_t head = static_cast<int32_t>(row_index / padded_rows);
        const int32_t packed_row = tiled_to_packed[padded_row];
        if (packed_row >= 0) {
            packed[(static_cast<std::int64_t>(head) * logical_rows + packed_row) * kHeadDim +
                   dimension] = tiled[index];
        }
    }
}

__global__ void mean_pool_tiles_kernel(const __nv_bfloat16* tiled, const int32_t* valid_sizes,
                                       float* pooled, int32_t total_tiles) {
    const int32_t head = blockIdx.y;
    const int32_t tile = blockIdx.x;
    const int32_t dimension = threadIdx.x;
    const int32_t valid = valid_sizes[tile];
    float sum = 0.0F;
    const std::int64_t tile_begin =
        (static_cast<std::int64_t>(head) * total_tiles + tile) * kTileTokens * kHeadDim;
    for (int32_t row = 0; row < valid; ++row)
        sum += __bfloat162float(tiled[tile_begin + row * kHeadDim + dimension]);
    pooled[(static_cast<std::int64_t>(head) * total_tiles + tile) * kHeadDim + dimension] =
        sum / static_cast<float>(valid);
}

__global__ void concatenate_valid_sizes_kernel(const int32_t* prefix_valid_sizes,
                                               const int32_t* video_valid_sizes,
                                               int32_t* valid_sizes, int32_t prefix_tiles,
                                               int32_t total_tiles) {
    for (int32_t index = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total_tiles; index += static_cast<int32_t>(gridDim.x) * blockDim.x) {
        valid_sizes[index] = index < prefix_tiles ? prefix_valid_sizes[index]
                                                  : video_valid_sizes[index - prefix_tiles];
    }
}

// The pooled tensors intentionally stay FP32, matching FastH3's selector and
// compression branch. A tiled kernel avoids making cuBLAS a model-plugin ABI
// or distribution dependency.
__global__ void pooled_qk_scores_kernel(const float* pooled_q, const float* pooled_k, float* scores,
                                        int32_t total_tiles) {
    __shared__ float query_tile[16][16];
    __shared__ float key_tile[16][16];

    const int32_t query = blockIdx.y * 16 + threadIdx.y;
    const int32_t key = blockIdx.x * 16 + threadIdx.x;
    const int32_t head = blockIdx.z;
    float result = 0.0F;
    for (int32_t begin = 0; begin < kHeadDim; begin += 16) {
        query_tile[threadIdx.y][threadIdx.x] =
            query < total_tiles
                ? pooled_q[(static_cast<std::int64_t>(head) * total_tiles + query) * kHeadDim +
                           begin + threadIdx.x]
                : 0.0F;
        key_tile[threadIdx.y][threadIdx.x] =
            key < total_tiles
                ? pooled_k[(static_cast<std::int64_t>(head) * total_tiles + key) * kHeadDim +
                           begin + threadIdx.y]
                : 0.0F;
        __syncthreads();
#pragma unroll
        for (int32_t offset = 0; offset < 16; ++offset)
            result = fmaf(query_tile[threadIdx.y][offset], key_tile[offset][threadIdx.x], result);
        __syncthreads();
    }
    if (query < total_tiles && key < total_tiles) {
        scores[(static_cast<std::int64_t>(head) * total_tiles + query) * total_tiles + key] =
            result * kAttentionScale;
    }
}

template <int32_t Threads, int32_t Items>
__global__ void select_video_topk_kernel(const float* scores, int32_t* selected,
                                         int32_t total_tiles, int32_t prefix_tiles,
                                         int32_t video_tiles, int32_t top_video_tiles) {
    using ScoreSort = cub::BlockRadixSort<float, Threads, Items, int32_t>;
    using IndexSort = cub::BlockRadixSort<int32_t, Threads, Items>;
    union SortStorage {
        typename ScoreSort::TempStorage score;
        typename IndexSort::TempStorage index;
    };
    __shared__ SortStorage storage;

    const int32_t query = blockIdx.x;
    const int32_t head = blockIdx.y;
    float keys[Items];
    int32_t indices[Items];
#pragma unroll
    for (int32_t item = 0; item < Items; ++item) {
        const int32_t video = threadIdx.x * Items + item;
        if (video < video_tiles) {
            const int32_t absolute_key = prefix_tiles + video;
            float score =
                scores[(static_cast<std::int64_t>(head) * total_tiles + query) * total_tiles +
                       absolute_key];
            keys[item] = isnan(score) ? -FLT_MAX : score;
            indices[item] = absolute_key;
        } else {
            keys[item] = -FLT_MAX;
            indices[item] = INT_MAX;
        }
    }
    // CUB's radix sort is stable, so equal finite scores retain the initial
    // ascending video index and match the CPU tie-break rule.
    ScoreSort(storage.score).SortDescending(keys, indices);
    __syncthreads();

#pragma unroll
    for (int32_t item = 0; item < Items; ++item) {
        const int32_t rank = threadIdx.x * Items + item;
        if (rank >= top_video_tiles)
            indices[item] = INT_MAX;
    }
    __syncthreads();
    IndexSort(storage.index).Sort(indices);
    __syncthreads();

#pragma unroll
    for (int32_t item = 0; item < Items; ++item) {
        const int32_t rank = threadIdx.x * Items + item;
        if (rank < top_video_tiles) {
            selected[(static_cast<std::int64_t>(head) * total_tiles + query) * top_video_tiles +
                     rank] = indices[item];
        }
    }
}

__device__ __forceinline__ int32_t attended_key_tile(int32_t key_rank, int32_t query_tile,
                                                     int32_t head, int32_t total_tiles,
                                                     int32_t prefix_tiles, int32_t top_video_tiles,
                                                     const int32_t* selected) {
    if (query_tile < prefix_tiles)
        return key_rank;
    if (key_rank < prefix_tiles)
        return key_rank;
    return selected[(static_cast<std::int64_t>(head) * total_tiles + query_tile) * top_video_tiles +
                    key_rank - prefix_tiles];
}

__global__ void block_sparse_attention_64_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    const int32_t* valid_sizes, const int32_t* selected_video_tiles, __nv_bfloat16* output,
    int32_t total_tiles, int32_t prefix_tiles, int32_t top_video_tiles) {
    extern __shared__ unsigned char shared_bytes[];
    auto* shared_q = reinterpret_cast<__nv_bfloat16*>(shared_bytes);
    auto* shared_scores = reinterpret_cast<float*>(shared_q + kTileTokens * kHeadDim);
    auto* shared_probabilities =
        reinterpret_cast<__nv_bfloat16*>(shared_scores + kTileTokens * kTileTokens);
    auto* shared_output =
        reinterpret_cast<float*>(shared_probabilities + kTileTokens * kTileTokens);
    auto* row_maximum = shared_output + kTileTokens * kHeadDim;
    auto* row_denominator = row_maximum + kTileTokens;

    const int32_t query_tile = blockIdx.x;
    const int32_t head = blockIdx.y;
    const int32_t warp = threadIdx.x / kWarpSize;
    const int32_t valid_query_rows = valid_sizes[query_tile];
    const int32_t key_count =
        query_tile < prefix_tiles ? total_tiles : prefix_tiles + top_video_tiles;
    const std::int64_t query_begin =
        (static_cast<std::int64_t>(head) * total_tiles + query_tile) * kTileTokens * kHeadDim;

    for (int32_t index = threadIdx.x; index < kTileTokens * kHeadDim; index += blockDim.x) {
        const int32_t row = index / kHeadDim;
        shared_q[index] =
            row < valid_query_rows ? query[query_begin + index] : __float2bfloat16_rn(0.0F);
        shared_output[index] = 0.0F;
    }
    if (threadIdx.x < kTileTokens) {
        row_maximum[threadIdx.x] = -FLT_MAX;
        row_denominator[threadIdx.x] = 0.0F;
    }
    __syncthreads();

    // Pass one finds a single stable-softmax maximum for every query row.
    for (int32_t key_rank = 0; key_rank < key_count; ++key_rank) {
        const int32_t key_tile =
            attended_key_tile(key_rank, query_tile, head, total_tiles, prefix_tiles,
                              top_video_tiles, selected_video_tiles);
        const int32_t valid_key_rows = valid_sizes[key_tile];
        const __nv_bfloat16* key_begin =
            key +
            (static_cast<std::int64_t>(head) * total_tiles + key_tile) * kTileTokens * kHeadDim;

#pragma unroll
        for (int32_t part = 0; part < 2; ++part) {
            const int32_t score_tile = warp + part * 8;
            const int32_t row_block = score_tile / 4;
            const int32_t column_block = score_tile % 4;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
            wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
            for (int32_t depth_block = 0; depth_block < 8; ++depth_block) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
                    query_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>
                    key_fragment;
                wmma::load_matrix_sync(query_fragment,
                                       shared_q + row_block * 16 * kHeadDim + depth_block * 16,
                                       kHeadDim);
                wmma::load_matrix_sync(key_fragment,
                                       key_begin + column_block * 16 * kHeadDim + depth_block * 16,
                                       kHeadDim);
                wmma::mma_sync(accumulator, query_fragment, key_fragment, accumulator);
            }
            wmma::store_matrix_sync(shared_scores + row_block * 16 * kTileTokens +
                                        column_block * 16,
                                    accumulator, kTileTokens, wmma::mem_row_major);
        }
        __syncthreads();
        if (threadIdx.x < valid_query_rows) {
            float maximum = row_maximum[threadIdx.x];
            const float* score_row = shared_scores + threadIdx.x * kTileTokens;
            for (int32_t column = 0; column < valid_key_rows; ++column)
                maximum = fmaxf(maximum, score_row[column] * kAttentionScale);
            row_maximum[threadIdx.x] = maximum;
        }
        __syncthreads();
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> output_accumulators[4];
#pragma unroll
    for (int32_t part = 0; part < 4; ++part)
        wmma::fill_fragment(output_accumulators[part], 0.0F);

    // Pass two recomputes QK, accumulates the unrounded denominator, casts
    // probabilities to BF16 at the published attention boundary, and applies V.
    for (int32_t key_rank = 0; key_rank < key_count; ++key_rank) {
        const int32_t key_tile =
            attended_key_tile(key_rank, query_tile, head, total_tiles, prefix_tiles,
                              top_video_tiles, selected_video_tiles);
        const int32_t valid_key_rows = valid_sizes[key_tile];
        const std::int64_t key_begin_offset =
            (static_cast<std::int64_t>(head) * total_tiles + key_tile) * kTileTokens * kHeadDim;
        const __nv_bfloat16* key_begin = key + key_begin_offset;
        const __nv_bfloat16* value_begin = value + key_begin_offset;

#pragma unroll
        for (int32_t part = 0; part < 2; ++part) {
            const int32_t score_tile = warp + part * 8;
            const int32_t row_block = score_tile / 4;
            const int32_t column_block = score_tile % 4;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
            wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
            for (int32_t depth_block = 0; depth_block < 8; ++depth_block) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
                    query_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>
                    key_fragment;
                wmma::load_matrix_sync(query_fragment,
                                       shared_q + row_block * 16 * kHeadDim + depth_block * 16,
                                       kHeadDim);
                wmma::load_matrix_sync(key_fragment,
                                       key_begin + column_block * 16 * kHeadDim + depth_block * 16,
                                       kHeadDim);
                wmma::mma_sync(accumulator, query_fragment, key_fragment, accumulator);
            }
            wmma::store_matrix_sync(shared_scores + row_block * 16 * kTileTokens +
                                        column_block * 16,
                                    accumulator, kTileTokens, wmma::mem_row_major);
        }
        __syncthreads();
        if (threadIdx.x < kTileTokens) {
            const int32_t row = threadIdx.x;
            float sum = 0.0F;
            for (int32_t column = 0; column < kTileTokens; ++column) {
                float probability = 0.0F;
                if (row < valid_query_rows && column < valid_key_rows) {
                    probability =
                        exp2f((shared_scores[row * kTileTokens + column] * kAttentionScale -
                               row_maximum[row]) *
                              kLog2E);
                    sum += probability;
                }
                shared_probabilities[row * kTileTokens + column] = __float2bfloat16_rn(probability);
            }
            row_denominator[row] += sum;
        }
        __syncthreads();

#pragma unroll
        for (int32_t part = 0; part < 4; ++part) {
            const int32_t output_tile = warp + part * 8;
            const int32_t row_block = output_tile / 8;
            const int32_t dimension_block = output_tile % 8;
#pragma unroll
            for (int32_t key_block = 0; key_block < 4; ++key_block) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
                    probability_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major>
                    value_fragment;
                wmma::load_matrix_sync(probability_fragment,
                                       shared_probabilities + row_block * 16 * kTileTokens +
                                           key_block * 16,
                                       kTileTokens);
                wmma::load_matrix_sync(
                    value_fragment, value_begin + key_block * 16 * kHeadDim + dimension_block * 16,
                    kHeadDim);
                wmma::mma_sync(output_accumulators[part], probability_fragment, value_fragment,
                               output_accumulators[part]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int32_t part = 0; part < 4; ++part) {
        const int32_t output_tile = warp + part * 8;
        const int32_t row_block = output_tile / 8;
        const int32_t dimension_block = output_tile % 8;
        wmma::store_matrix_sync(shared_output + row_block * 16 * kHeadDim + dimension_block * 16,
                                output_accumulators[part], kHeadDim, wmma::mem_row_major);
    }
    __syncthreads();

    for (int32_t index = threadIdx.x; index < kTileTokens * kHeadDim; index += blockDim.x) {
        const int32_t row = index / kHeadDim;
        const float denominator = row_denominator[row];
        output[query_begin + index] = row < valid_query_rows && denominator > 0.0F
                                          ? __float2bfloat16_rn(shared_output[index] / denominator)
                                          : __float2bfloat16_rn(0.0F);
    }
}

__global__ void pooled_gate_attention_kernel(const float* scores, const float* pooled_v,
                                             float* compressed, int32_t total_tiles) {
    __shared__ float reduction[128];
    const int32_t query = blockIdx.x;
    const int32_t head = blockIdx.y;
    const int32_t dimension = threadIdx.x;
    const float* score_row =
        scores + (static_cast<std::int64_t>(head) * total_tiles + query) * total_tiles;

    float local_maximum = -FLT_MAX;
    for (int32_t key = dimension; key < total_tiles; key += blockDim.x)
        local_maximum = fmaxf(local_maximum, score_row[key]);
    reduction[dimension] = local_maximum;
    __syncthreads();
    for (int32_t width = 64; width > 0; width /= 2) {
        if (dimension < width)
            reduction[dimension] = fmaxf(reduction[dimension], reduction[dimension + width]);
        __syncthreads();
    }
    const float maximum = reduction[0];

    float local_sum = 0.0F;
    for (int32_t key = dimension; key < total_tiles; key += blockDim.x)
        local_sum += exp2f((score_row[key] - maximum) * kLog2E);
    reduction[dimension] = local_sum;
    __syncthreads();
    for (int32_t width = 64; width > 0; width /= 2) {
        if (dimension < width)
            reduction[dimension] += reduction[dimension + width];
        __syncthreads();
    }
    const float denominator = reduction[0];

    float result = 0.0F;
    for (int32_t key = 0; key < total_tiles; ++key) {
        const float probability = exp2f((score_row[key] - maximum) * kLog2E) / denominator;
        result = fmaf(
            probability,
            pooled_v[(static_cast<std::int64_t>(head) * total_tiles + key) * kHeadDim + dimension],
            result);
    }
    compressed[(static_cast<std::int64_t>(head) * total_tiles + query) * kHeadDim + dimension] =
        result;
}

__global__ void merge_gate_kernel(const __nv_bfloat16* sparse, const __nv_bfloat16* gate,
                                  const float* compressed, __nv_bfloat16* output,
                                  std::int64_t element_count, int32_t total_tiles) {
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < element_count; index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int32_t dimension = static_cast<int32_t>(index % kHeadDim);
        const std::int64_t row = index / kHeadDim;
        const std::int64_t head = row / (total_tiles * kTileTokens);
        const int32_t padded_row = static_cast<int32_t>(row % (total_tiles * kTileTokens));
        const int32_t tile_in_head = padded_row / kTileTokens;
        const float value =
            __bfloat162float(sparse[index]) +
            __bfloat162float(gate[index]) *
                compressed[(head * total_tiles + tile_in_head) * kHeadDim + dimension];
        output[index] = __float2bfloat16_rn(value);
    }
}

} // namespace

void tile_bhsd_async(const __nv_bfloat16* packed, const int32_t* tiled_to_packed,
                     __nv_bfloat16* tiled, int32_t heads, int32_t logical_rows, int32_t total_tiles,
                     cudaStream_t stream) {
    validate_tiled(packed, tiled, heads, logical_rows, total_tiles);
    if (tiled_to_packed == nullptr)
        throw std::invalid_argument("FastH3 VSA tile map must not be null");
    constexpr int32_t threads = 256;
    const std::int64_t count =
        static_cast<std::int64_t>(heads) * total_tiles * kTileTokens * kHeadDim;
    const int32_t blocks =
        static_cast<int32_t>(std::min<std::int64_t>((count + threads - 1) / threads, 65535));
    tile_bhsd_kernel<<<blocks, threads, 0, stream>>>(packed, tiled_to_packed, tiled, logical_rows,
                                                     total_tiles * kTileTokens, heads);
    check_launch("FastH3 VSA tile launch");
}

void untile_bhsd_async(const __nv_bfloat16* tiled, const int32_t* tiled_to_packed,
                       __nv_bfloat16* packed, int32_t heads, int32_t logical_rows,
                       int32_t total_tiles, cudaStream_t stream) {
    validate_tiled(tiled, packed, heads, logical_rows, total_tiles);
    if (tiled_to_packed == nullptr)
        throw std::invalid_argument("FastH3 VSA untile map must not be null");
    constexpr int32_t threads = 256;
    const std::int64_t count =
        static_cast<std::int64_t>(heads) * total_tiles * kTileTokens * kHeadDim;
    const int32_t blocks =
        static_cast<int32_t>(std::min<std::int64_t>((count + threads - 1) / threads, 65535));
    untile_bhsd_kernel<<<blocks, threads, 0, stream>>>(tiled, tiled_to_packed, packed, logical_rows,
                                                       total_tiles * kTileTokens, heads);
    check_launch("FastH3 VSA untile launch");
}

void mean_pool_tiles_async(const __nv_bfloat16* tiled, const int32_t* valid_sizes, float* pooled,
                           int32_t heads, int32_t total_tiles, cudaStream_t stream) {
    if (tiled == nullptr || valid_sizes == nullptr || pooled == nullptr || heads <= 0 ||
        total_tiles <= 0)
        throw std::invalid_argument("FastH3 VSA mean-pool launch received invalid arguments");
    mean_pool_tiles_kernel<<<dim3(total_tiles, heads), kHeadDim, 0, stream>>>(tiled, valid_sizes,
                                                                              pooled, total_tiles);
    check_launch("FastH3 VSA mean-pool launch");
}

void concatenate_valid_sizes_async(const int32_t* prefix_valid_sizes,
                                   const int32_t* video_valid_sizes, int32_t* valid_sizes,
                                   int32_t prefix_tiles, int32_t video_tiles, cudaStream_t stream) {
    if (prefix_valid_sizes == nullptr || video_valid_sizes == nullptr || valid_sizes == nullptr ||
        prefix_tiles <= 0 || video_tiles <= 0 || video_tiles > kMaxVideoTiles) {
        throw std::invalid_argument(
            "FastH3 VSA valid-size concatenation received invalid arguments");
    }
    constexpr int32_t threads = 256;
    const int32_t total_tiles = prefix_tiles + video_tiles;
    const int32_t blocks = (total_tiles + threads - 1) / threads;
    concatenate_valid_sizes_kernel<<<blocks, threads, 0, stream>>>(
        prefix_valid_sizes, video_valid_sizes, valid_sizes, prefix_tiles, total_tiles);
    check_launch("FastH3 VSA valid-size concatenation launch");
}

void pooled_qk_scores_async(const float* pooled_q, const float* pooled_k, float* scores,
                            int32_t heads, int32_t total_tiles, cudaStream_t stream) {
    if (pooled_q == nullptr || pooled_k == nullptr || scores == nullptr || heads <= 0 ||
        total_tiles <= 0)
        throw std::invalid_argument("FastH3 VSA pooled-QK launch received invalid arguments");
    pooled_qk_scores_kernel<<<dim3((total_tiles + 15) / 16, (total_tiles + 15) / 16, heads),
                              dim3(16, 16), 0, stream>>>(pooled_q, pooled_k, scores, total_tiles);
    check_launch("FastH3 VSA pooled-QK launch");
}

void select_video_topk_async(const float* scores, int32_t* selected_video_tiles, int32_t heads,
                             int32_t total_tiles, int32_t prefix_tiles, int32_t video_tiles,
                             int32_t top_video_tiles, cudaStream_t stream) {
    validate_sparse_geometry(heads, total_tiles, prefix_tiles, video_tiles, top_video_tiles);
    if (scores == nullptr || selected_video_tiles == nullptr || video_tiles > kSelectorCapacity)
        throw std::invalid_argument("FastH3 VSA selector received invalid arguments");
    select_video_topk_kernel<kSelectorThreads, kSelectorItems>
        <<<dim3(total_tiles, heads), kSelectorThreads, 0, stream>>>(
            scores, selected_video_tiles, total_tiles, prefix_tiles, video_tiles, top_video_tiles);
    check_launch("FastH3 VSA selector launch");
}

void block_sparse_attention_64_async(const __nv_bfloat16* query, const __nv_bfloat16* key,
                                     const __nv_bfloat16* value, const int32_t* valid_sizes,
                                     const int32_t* selected_video_tiles, __nv_bfloat16* output,
                                     int32_t heads, int32_t total_tiles, int32_t prefix_tiles,
                                     int32_t video_tiles, int32_t top_video_tiles,
                                     cudaStream_t stream) {
    validate_sparse_geometry(heads, total_tiles, prefix_tiles, video_tiles, top_video_tiles);
    if (query == nullptr || key == nullptr || value == nullptr || valid_sizes == nullptr ||
        selected_video_tiles == nullptr || output == nullptr)
        throw std::invalid_argument("FastH3 VSA attention launch received null tensors");
    constexpr std::size_t shared_bytes =
        kTileTokens * kHeadDim * sizeof(__nv_bfloat16) + kTileTokens * kTileTokens * sizeof(float) +
        kTileTokens * kTileTokens * sizeof(__nv_bfloat16) + kTileTokens * kHeadDim * sizeof(float) +
        2 * kTileTokens * sizeof(float);
    check_cuda(cudaFuncSetAttribute(block_sparse_attention_64_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int32_t>(shared_bytes)),
               "FastH3 VSA attention shared-memory opt-in");
    block_sparse_attention_64_kernel<<<dim3(total_tiles, heads), kAttentionThreads, shared_bytes,
                                       stream>>>(query, key, value, valid_sizes,
                                                 selected_video_tiles, output, total_tiles,
                                                 prefix_tiles, top_video_tiles);
    check_launch("FastH3 VSA attention launch");
}

void pooled_gate_attention_async(const float* scores, const float* pooled_v, float* compressed,
                                 int32_t heads, int32_t total_tiles, cudaStream_t stream) {
    if (scores == nullptr || pooled_v == nullptr || compressed == nullptr || heads <= 0 ||
        total_tiles <= 0)
        throw std::invalid_argument("FastH3 VSA gate-attention launch received invalid arguments");
    pooled_gate_attention_kernel<<<dim3(total_tiles, heads), kHeadDim, 0, stream>>>(
        scores, pooled_v, compressed, total_tiles);
    check_launch("FastH3 VSA gate-attention launch");
}

void merge_gate_async(const __nv_bfloat16* sparse, const __nv_bfloat16* gate,
                      const float* compressed, __nv_bfloat16* output, int32_t heads,
                      int32_t total_tiles, cudaStream_t stream) {
    if (sparse == nullptr || gate == nullptr || compressed == nullptr || output == nullptr ||
        heads <= 0 || total_tiles <= 0)
        throw std::invalid_argument("FastH3 VSA gate-merge launch received invalid arguments");
    constexpr int32_t threads = 256;
    const std::int64_t count =
        static_cast<std::int64_t>(heads) * total_tiles * kTileTokens * kHeadDim;
    const int32_t blocks =
        static_cast<int32_t>(std::min<std::int64_t>((count + threads - 1) / threads, 65535));
    merge_gate_kernel<<<blocks, threads, 0, stream>>>(sparse, gate, compressed, output, count,
                                                      total_tiles);
    check_launch("FastH3 VSA gate-merge launch");
}

} // namespace trtmc::minimax_h3::vsa
