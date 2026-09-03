/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/vsa_attention.h"

#ifndef TRTMC_MINIMAX_H3_HAS_SM121_SPECIALIZATION
#define TRTMC_MINIMAX_H3_HAS_SM121_SPECIALIZATION 0
#endif

#if TRTMC_MINIMAX_H3_HAS_SM121_SPECIALIZATION && CUDART_VERSION < 12090
#error "The MiniMax-H3 SM121 specialization requires CUDA Runtime headers 12.9 or newer"
#endif

#if TRTMC_MINIMAX_H3_HAS_SM121_SPECIALIZATION
#include "vsa_attention_sm121_cubin.h"
#endif

#include <algorithm>
#if TRTMC_MINIMAX_H3_HAS_SM121_SPECIALIZATION
#include <array>
#endif
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <limits>
#if TRTMC_MINIMAX_H3_HAS_SM121_SPECIALIZATION
#include <mutex>
#endif
#include <mma.h>
#include <stdexcept>
#include <string>

namespace trtmc::minimax_h3::vsa {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int32_t kWarpSize = 32;
constexpr int32_t kAttentionThreads = 128;
constexpr int32_t kAttentionQueryRows = 16;
constexpr int32_t kAttentionQuerySlices = kTileTokens / kAttentionQueryRows;
constexpr int32_t kSelectorThreads = 256;
// The public worst-aspect 15-second canvas has 2,080 video tiles. Nine
// blocked items per thread keep the one-block radix selector fail-closed above
// that profile while covering all released geometry (capacity 2,304).
constexpr int32_t kSelectorItems = 9;
constexpr int32_t kSelectorCapacity = kSelectorThreads * kSelectorItems;
constexpr float kAttentionScale = 0.08838834764831844055F; // 1 / sqrt(128)
constexpr float kLog2E = 1.4426950408889634074F;
#if TRTMC_MINIMAX_H3_HAS_SM121_SPECIALIZATION
constexpr int32_t kSm121Threads = 128;
constexpr int32_t kSm121SharedBytes = 90136;

struct Sm121KernelState {
    std::mutex mutex;
    cudaLibrary_t library{nullptr};
    cudaKernel_t kernel{nullptr};
    cudaError_t library_failure{cudaSuccess};
    bool attempted{false};
    // 0: unconfigured, 1: ready, 2: failed. CUDA supports at most 32 devices.
    std::array<unsigned char, 32> device_status{};
    std::array<cudaError_t, 32> device_query_failure{};
    std::array<cudaError_t, 32> device_configuration_failure{};
};

thread_local cudaError_t sm121_thread_query_failure = cudaSuccess;

Sm121KernelState& sm121_kernel_state() {
    // The CUDA Runtime owns loaded library lifetime. Keeping this state until
    // process exit avoids unloading code while another stream may still use it.
    static auto* state = new Sm121KernelState();
    return *state;
}

bool query_sm121_device(int32_t& device, bool& supported) {
    device = -1;
    supported = false;
    int32_t major = 0;
    int32_t minor = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        sm121_thread_query_failure = status;
        cudaGetLastError();
        return false;
    }
    if (device < 0 || device >= 32) {
        sm121_thread_query_failure = cudaErrorInvalidDevice;
        return false;
    }

    status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (status == cudaSuccess)
        status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    {
        auto& state = sm121_kernel_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.device_query_failure[static_cast<std::size_t>(device)] = status;
    }
    sm121_thread_query_failure = status;
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    supported = major == 12 && minor == 1;
    return true;
}

cudaKernel_t sm121_kernel_for_device(int32_t device) {
    auto& state = sm121_kernel_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.attempted) {
        state.attempted = true;
        state.library_failure = cudaLibraryLoadData(&state.library, kMiniMaxH3VsaSm121Cubin,
                                                    nullptr, nullptr, 0, nullptr, nullptr, 0);
        if (state.library_failure == cudaSuccess) {
            state.library_failure =
                cudaLibraryGetKernel(&state.kernel, state.library, "_attn_fwd_sparse");
        }
        if (state.library_failure != cudaSuccess) {
            state.kernel = nullptr;
            cudaGetLastError();
        }
    }
    if (state.kernel == nullptr || state.device_status[static_cast<std::size_t>(device)] == 2)
        return nullptr;
    if (state.device_status[static_cast<std::size_t>(device)] == 0) {
        const cudaError_t status = cudaKernelSetAttributeForDevice(
            state.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSm121SharedBytes, device);
        state.device_configuration_failure[static_cast<std::size_t>(device)] = status;
        state.device_status[static_cast<std::size_t>(device)] = status == cudaSuccess ? 1 : 2;
        if (status != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
    }
    return state.kernel;
}

cudaKernel_t sm121_kernel() {
    int32_t device = -1;
    bool supported = false;
    if (!query_sm121_device(device, supported) || !supported)
        return nullptr;
    return sm121_kernel_for_device(device);
}
#endif

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

__device__ bool finite_value(__nv_bfloat16 value) {
    return isfinite(__bfloat162float(value));
}

__device__ bool finite_value(float value) {
    return isfinite(value);
}

template <typename Value>
__global__ void all_finite_kernel(const Value* values, std::size_t count, uint32_t* all_finite) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        if (!finite_value(values[index])) {
            atomicExch(all_finite, 0U);
            return;
        }
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

#if TRTMC_MINIMAX_H3_HAS_SM121_SPECIALIZATION
// The embedded SM121 specialization uses FastVideo's generic q2k ABI. Expand
// ModelConnect's compact selector output into that ABI without materializing a
// dense Boolean mask. Only live entries are written; unused row tails are never
// read by the attention kernel.
__global__ void materialize_sm121_q2k_kernel(const int32_t* selected_video_tiles,
                                             int32_t* q2k_index, int32_t* q2k_count, int32_t heads,
                                             int32_t total_tiles, int32_t prefix_tiles,
                                             int32_t top_video_tiles) {
    const int32_t row = blockIdx.x;
    const int32_t query_tile = row % total_tiles;
    const int32_t head = row / total_tiles;
    if (head >= heads)
        return;
    const bool dense = query_tile < prefix_tiles;
    const int32_t count = dense ? total_tiles : prefix_tiles + top_video_tiles;
    if (threadIdx.x == 0)
        q2k_count[row] = count;
    for (int32_t rank = threadIdx.x; rank < count; rank += blockDim.x) {
        int32_t key_tile = rank;
        if (!dense && rank >= prefix_tiles) {
            key_tile =
                selected_video_tiles[(static_cast<std::int64_t>(head) * total_tiles + query_tile) *
                                         top_video_tiles +
                                     rank - prefix_tiles];
        }
        q2k_index[static_cast<std::int64_t>(row) * total_tiles + rank] = key_tile;
    }
}

__global__ void zero_sm121_padded_query_rows_kernel(__nv_bfloat16* output,
                                                    const int32_t* valid_sizes,
                                                    int32_t total_tiles) {
    const int32_t row = blockIdx.x;
    const int32_t query_tile = row % total_tiles;
    const int32_t head = row / total_tiles;
    const int32_t valid_rows = valid_sizes[query_tile];
    for (int32_t index = valid_rows * kHeadDim + threadIdx.x; index < kTileTokens * kHeadDim;
         index += blockDim.x) {
        output[(static_cast<std::int64_t>(head) * total_tiles + query_tile) * kTileTokens *
                   kHeadDim +
               index] = __float2bfloat16_rn(0.0F);
    }
}

void launch_sm121_attention(cudaKernel_t kernel, const __nv_bfloat16* query,
                            const __nv_bfloat16* key, const __nv_bfloat16* value,
                            const int32_t* valid_sizes, const int32_t* selected_video_tiles,
                            __nv_bfloat16* output, int32_t* q2k_index, int32_t* q2k_count,
                            float* lse, int32_t heads, int32_t total_tiles, int32_t prefix_tiles,
                            int32_t top_video_tiles, cudaStream_t stream) {
    if (kernel == nullptr || q2k_index == nullptr || q2k_count == nullptr || lse == nullptr)
        throw std::invalid_argument("FastH3 SM121 attention received invalid workspace");

    materialize_sm121_q2k_kernel<<<heads * total_tiles, 256, 0, stream>>>(
        selected_video_tiles, q2k_index, q2k_count, heads, total_tiles, prefix_tiles,
        top_video_tiles);
    check_launch("FastH3 SM121 q2k materialization launch");

    auto* query_arg = const_cast<__nv_bfloat16*>(query);
    auto* key_arg = const_cast<__nv_bfloat16*>(key);
    auto* value_arg = const_cast<__nv_bfloat16*>(value);
    float scale = kAttentionScale;
    auto* q2k_index_arg = q2k_index;
    auto* q2k_count_arg = q2k_count;
    int32_t max_kv_blocks = total_tiles;
    auto* valid_sizes_arg = const_cast<int32_t*>(valid_sizes);
    auto* lse_arg = lse;
    auto* output_arg = output;
    const int32_t tokens = total_tiles * kTileTokens;
    int32_t stride_z = heads * tokens * kHeadDim;
    int32_t stride_h = tokens * kHeadDim;
    int32_t stride_m = kHeadDim;
    int32_t heads_arg = heads;
    int32_t query_tokens = tokens;
    int32_t key_tokens = tokens;
    void* global_scratch = nullptr;
    void* profile_scratch = nullptr;
    void* arguments[] = {
        &query_arg,      &key_arg,         &value_arg,       &scale,        &q2k_index_arg,
        &q2k_count_arg,  &max_kv_blocks,   &valid_sizes_arg, &lse_arg,      &output_arg,
        &stride_z,       &stride_h,        &stride_m,        &stride_z,     &stride_h,
        &stride_m,       &stride_z,        &stride_h,        &stride_m,     &stride_z,
        &stride_h,       &stride_m,        &heads_arg,       &query_tokens, &key_tokens,
        &global_scratch, &profile_scratch,
    };
    check_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(total_tiles, heads, 1),
                                dim3(kSm121Threads, 1, 1), arguments, kSm121SharedBytes, stream),
               "FastH3 SM121 online-attention launch");
    zero_sm121_padded_query_rows_kernel<<<heads * total_tiles, kSm121Threads, 0, stream>>>(
        output, valid_sizes, total_tiles);
    check_launch("FastH3 SM121 padding cleanup launch");
}
#endif

__global__ void block_sparse_attention_64_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    const int32_t* valid_sizes, const int32_t* selected_video_tiles, __nv_bfloat16* output,
    int32_t total_tiles, int32_t prefix_tiles, int32_t top_video_tiles) {
    extern __shared__ unsigned char shared_bytes[];
    auto* shared_q = reinterpret_cast<__nv_bfloat16*>(shared_bytes);
    auto* shared_scores = reinterpret_cast<float*>(shared_q + kAttentionQueryRows * kHeadDim);
    auto* shared_probabilities =
        reinterpret_cast<__nv_bfloat16*>(shared_scores + kAttentionQueryRows * kTileTokens);
    // Each warp needs one 16x16 FP32 landing tile only while converting its
    // accumulator to BF16. Reusing it across the two output-column fragments
    // avoids a full 64x128 FP32 shared-output allocation.
    auto* shared_warp_store =
        reinterpret_cast<float*>(shared_probabilities + kAttentionQueryRows * kTileTokens);
    auto* row_maximum = shared_warp_store + (kAttentionThreads / kWarpSize) * 16 * 16;
    auto* row_denominator = row_maximum + kAttentionQueryRows;

    const int32_t query_tile = blockIdx.x;
    const int32_t head = blockIdx.y;
    const int32_t query_slice = blockIdx.z;
    const int32_t query_row_offset = query_slice * kAttentionQueryRows;
    const int32_t warp = threadIdx.x / kWarpSize;
    const int32_t lane = threadIdx.x % kWarpSize;
    const int32_t remaining_query_rows = valid_sizes[query_tile] - query_row_offset;
    const int32_t valid_query_rows =
        remaining_query_rows <= 0
            ? 0
            : (remaining_query_rows < kAttentionQueryRows ? remaining_query_rows
                                                          : kAttentionQueryRows);
    const int32_t key_count =
        query_tile < prefix_tiles ? total_tiles : prefix_tiles + top_video_tiles;
    const std::int64_t query_begin =
        (static_cast<std::int64_t>(head) * total_tiles + query_tile) * kTileTokens * kHeadDim +
        query_row_offset * kHeadDim;

    for (int32_t index = threadIdx.x; index < kAttentionQueryRows * kHeadDim; index += blockDim.x) {
        const int32_t row = index / kHeadDim;
        shared_q[index] =
            row < valid_query_rows ? query[query_begin + index] : __float2bfloat16_rn(0.0F);
    }
    if (threadIdx.x < kAttentionQueryRows) {
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

        const int32_t column_block = warp;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
        wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
        for (int32_t depth_block = 0; depth_block < 8; ++depth_block) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
                query_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> key_fragment;
            wmma::load_matrix_sync(query_fragment, shared_q + depth_block * 16, kHeadDim);
            wmma::load_matrix_sync(key_fragment,
                                   key_begin + column_block * 16 * kHeadDim + depth_block * 16,
                                   kHeadDim);
            wmma::mma_sync(accumulator, query_fragment, key_fragment, accumulator);
        }
        wmma::store_matrix_sync(shared_scores + column_block * 16, accumulator, kTileTokens,
                                wmma::mem_row_major);
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

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> output_accumulators[2];
#pragma unroll
    for (int32_t part = 0; part < 2; ++part)
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

        const int32_t column_block = warp;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
        wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
        for (int32_t depth_block = 0; depth_block < 8; ++depth_block) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
                query_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> key_fragment;
            wmma::load_matrix_sync(query_fragment, shared_q + depth_block * 16, kHeadDim);
            wmma::load_matrix_sync(key_fragment,
                                   key_begin + column_block * 16 * kHeadDim + depth_block * 16,
                                   kHeadDim);
            wmma::mma_sync(accumulator, query_fragment, key_fragment, accumulator);
        }
        wmma::store_matrix_sync(shared_scores + column_block * 16, accumulator, kTileTokens,
                                wmma::mem_row_major);
        __syncthreads();
        if (threadIdx.x < kAttentionQueryRows) {
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
        for (int32_t part = 0; part < 2; ++part) {
            const int32_t dimension_block = warp + part * 4;
#pragma unroll
            for (int32_t key_block = 0; key_block < 4; ++key_block) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
                    probability_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major>
                    value_fragment;
                wmma::load_matrix_sync(probability_fragment, shared_probabilities + key_block * 16,
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
    for (int32_t part = 0; part < 2; ++part) {
        const int32_t dimension_block = warp + part * 4;
        float* warp_store = shared_warp_store + warp * 16 * 16;
        wmma::store_matrix_sync(warp_store, output_accumulators[part], 16, wmma::mem_row_major);
        __syncwarp();
        for (int32_t index = lane; index < 16 * 16; index += kWarpSize) {
            const int32_t row = index / 16;
            const int32_t column = index % 16;
            const float denominator = row_denominator[row];
            output[query_begin + row * kHeadDim + dimension_block * 16 + column] =
                row < valid_query_rows && denominator > 0.0F
                    ? __float2bfloat16_rn(warp_store[index] / denominator)
                    : __float2bfloat16_rn(0.0F);
        }
        __syncwarp();
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

template <typename Value>
bool all_finite_sync(const Value* values, std::size_t count, uint32_t* workspace,
                     std::size_t workspace_capacity, cudaStream_t stream) {
    if (values == nullptr || workspace == nullptr || count == 0 ||
        workspace_capacity < kAllFiniteWorkspaceWords) {
        throw std::invalid_argument("FastH3 finite-output scan received invalid arguments");
    }
    check_cuda(cudaMemsetAsync(workspace, 0xff, sizeof(uint32_t), stream),
               "FastH3 finite-output reset");
    constexpr std::size_t threads = 256;
    constexpr std::size_t maximum_blocks = 4096;
    const auto blocks = static_cast<uint32_t>(
        std::min<std::size_t>((count + threads - 1) / threads, maximum_blocks));
    all_finite_kernel<<<blocks, threads, 0, stream>>>(values, count, workspace);
    check_launch("FastH3 finite-output scan launch");
    uint32_t host = 0;
    check_cuda(cudaMemcpyAsync(&host, workspace, sizeof(host), cudaMemcpyDeviceToHost, stream),
               "FastH3 finite-output download");
    check_cuda(cudaStreamSynchronize(stream), "FastH3 finite-output synchronize");
    return host != 0;
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

Sm121AttentionStatus block_sparse_attention_sm121_status() {
#if TRTMC_MINIMAX_H3_HAS_SM121_SPECIALIZATION
    int32_t device = -1;
    bool supported = false;
    if (!query_sm121_device(device, supported))
        return Sm121AttentionStatus::kLoadFailed;
    if (!supported)
        return Sm121AttentionStatus::kUnsupportedDevice;
    return sm121_kernel_for_device(device) != nullptr ? Sm121AttentionStatus::kReady
                                                      : Sm121AttentionStatus::kLoadFailed;
#else
    return Sm121AttentionStatus::kNotBuilt;
#endif
}

cudaError_t block_sparse_attention_sm121_failure() {
#if TRTMC_MINIMAX_H3_HAS_SM121_SPECIALIZATION
    int32_t device = -1;
    const cudaError_t current_device_status = cudaGetDevice(&device);
    if (current_device_status != cudaSuccess) {
        cudaGetLastError();
        return current_device_status;
    }
    if (device < 0 || device >= 32)
        return cudaErrorInvalidDevice;

    auto& state = sm121_kernel_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.library_failure != cudaSuccess)
        return state.library_failure;
    const auto query_failure = state.device_query_failure[static_cast<std::size_t>(device)];
    if (query_failure != cudaSuccess)
        return query_failure;
    const auto configuration_failure =
        state.device_configuration_failure[static_cast<std::size_t>(device)];
    if (configuration_failure != cudaSuccess)
        return configuration_failure;
    return sm121_thread_query_failure;
#else
    return cudaErrorNotSupported;
#endif
}

bool block_sparse_attention_sm121_available() {
    return block_sparse_attention_sm121_status() == Sm121AttentionStatus::kReady;
}

bool bfloat16_all_finite_sync(const __nv_bfloat16* values, std::size_t count, uint32_t* workspace,
                              std::size_t workspace_capacity, cudaStream_t stream) {
    return all_finite_sync(values, count, workspace, workspace_capacity, stream);
}

bool float_all_finite_sync(const float* values, std::size_t count, uint32_t* workspace,
                           std::size_t workspace_capacity, cudaStream_t stream) {
    return all_finite_sync(values, count, workspace, workspace_capacity, stream);
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
    constexpr std::size_t shared_bytes = kAttentionQueryRows * kHeadDim * sizeof(__nv_bfloat16) +
                                         kAttentionQueryRows * kTileTokens * sizeof(float) +
                                         kAttentionQueryRows * kTileTokens * sizeof(__nv_bfloat16) +
                                         (kAttentionThreads / kWarpSize) * 16 * 16 * sizeof(float) +
                                         2 * kAttentionQueryRows * sizeof(float);
    check_cuda(cudaFuncSetAttribute(block_sparse_attention_64_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int32_t>(shared_bytes)),
               "FastH3 VSA attention shared-memory opt-in");
    block_sparse_attention_64_kernel<<<dim3(total_tiles, heads, kAttentionQuerySlices),
                                       kAttentionThreads, shared_bytes, stream>>>(
        query, key, value, valid_sizes, selected_video_tiles, output, total_tiles, prefix_tiles,
        top_video_tiles);
    check_launch("FastH3 VSA attention launch");
}

void block_sparse_attention_64_sm121_async(
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    const int32_t* valid_sizes, const int32_t* selected_video_tiles, __nv_bfloat16* output,
    int32_t heads, int32_t total_tiles, int32_t prefix_tiles, int32_t video_tiles,
    int32_t top_video_tiles, cudaStream_t stream, const Sm121AttentionWorkspace& workspace) {
    validate_sparse_geometry(heads, total_tiles, prefix_tiles, video_tiles, top_video_tiles);
    if (query == nullptr || key == nullptr || value == nullptr || valid_sizes == nullptr ||
        selected_video_tiles == nullptr || output == nullptr)
        throw std::invalid_argument("FastH3 SM121 attention launch received null tensors");
    const auto rows = static_cast<std::size_t>(heads) * total_tiles;
    const auto required_q2k_index = rows * total_tiles;
    const auto required_lse = rows * kTileTokens;
    if (workspace.q2k_index == nullptr || workspace.q2k_count == nullptr ||
        workspace.lse == nullptr || workspace.q2k_index_capacity < required_q2k_index ||
        workspace.q2k_count_capacity < rows || workspace.lse_capacity < required_lse) {
        throw std::invalid_argument("FastH3 SM121 attention workspace is too small");
    }
#if TRTMC_MINIMAX_H3_HAS_SM121_SPECIALIZATION
    auto kernel = sm121_kernel();
    if (kernel == nullptr)
        throw std::runtime_error("FastH3 SM121 attention specialization is unavailable");
    launch_sm121_attention(kernel, query, key, value, valid_sizes, selected_video_tiles, output,
                           workspace.q2k_index, workspace.q2k_count, workspace.lse, heads,
                           total_tiles, prefix_tiles, top_video_tiles, stream);
#else
    static_cast<void>(stream);
    throw std::runtime_error("FastH3 SM121 attention specialization was not built");
#endif
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
