/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugins/runtime_kv/cudnn_attention.h"
#include "plugins/runtime_kv/native_contiguous_attention_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <limits>
#include <new>

namespace trtmc::runtime_kv {
namespace {

constexpr int32_t kAttentionThreads = 256;
constexpr int32_t kAttentionWarps = kAttentionThreads / 32;
constexpr int32_t kFusedDecodeTileRows = 128;
// The direct Sq=1 path is deliberately a small/medium-history specialization.
// Larger bound extents retain cuDNN SDPA so full-context decode does not trade
// away parallelism merely to reduce launch overhead at the first buckets.
constexpr int32_t kFusedDecodeHistoryRowsLimit = 512;
constexpr std::size_t kFusedDecodeSharedMemoryLimit = 32U << 10;
constexpr std::size_t kWorkspaceAlignment = 256;

std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    if (alignment == 0 || value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        return 0;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

__device__ bool valid_request(int32_t history_length, int32_t history_rows, int32_t query_rows,
                              int32_t chunk_limit) {
    bool const valid_cold = history_length == 0 && history_rows == 1;
    bool const valid_non_cold =
        history_length > 0 && history_rows >= 2 && history_length <= history_rows;
    return query_rows > 0 && query_rows <= chunk_limit && (valid_cold || valid_non_cold);
}

__global__ void prepare_sequence_lengths(int32_t const* history_length_ptr,
                                         int32_t* sequence_length_q,
                                         int32_t* sequence_length_history,
                                         int32_t* sequence_length_current, int32_t* request_valid,
                                         int32_t history_rows, int32_t query_rows,
                                         int32_t chunk_limit) {
    int32_t const history_length = *history_length_ptr;
    if (valid_request(history_length, history_rows, query_rows, chunk_limit)) {
        *request_valid = 1;
        *sequence_length_q = query_rows;
        *sequence_length_history = history_length;
        *sequence_length_current = query_rows;
    } else {
        // Preserve memory safety if a caller bypasses runtime admission.
        // Zero lengths prevent cuDNN from consuming rows outside T.
        *request_valid = 0;
        *sequence_length_q = 0;
        *sequence_length_history = 0;
        *sequence_length_current = 0;
    }
}

__global__ void zero_context_if_invalid(int32_t const* request_valid, __nv_bfloat16* context,
                                        int64_t elements) {
    if (*request_valid != 0) {
        return;
    }
    int64_t const element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element < elements) {
        context[element] = __float2bfloat16_rn(0.0F);
    }
}

__global__ void pack_padded_head_major(__nv_bfloat16 const* source, __nv_bfloat16* destination,
                                       int32_t rows, int32_t padded_rows, int32_t num_heads,
                                       int32_t head_dim) {
    int64_t const element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const count = static_cast<int64_t>(num_heads) * rows * head_dim;
    if (element >= count) {
        return;
    }
    int32_t const dim = element % head_dim;
    int64_t const row_and_head = element / head_dim;
    int32_t const row = row_and_head % rows;
    int32_t const head = row_and_head / rows;
    int64_t const destination_index =
        (static_cast<int64_t>(head) * padded_rows + row) * head_dim + dim;
    destination[destination_index] = source[element];
}

__global__ void combine_segmented_context(
    __nv_bfloat16 const* history_context, __nv_bfloat16 const* current_context,
    float const* history_log_sum_exp, float const* current_log_sum_exp, __nv_bfloat16* destination,
    int32_t query_rows, int32_t padded_query_rows, int32_t num_query_heads, int32_t head_dim) {
    int64_t const element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const count = static_cast<int64_t>(num_query_heads) * query_rows * head_dim;
    if (element >= count) {
        return;
    }
    int32_t const dim = element % head_dim;
    int64_t const row_and_head = element / head_dim;
    int32_t const row = row_and_head % query_rows;
    int32_t const head = row_and_head / query_rows;
    int64_t const source_index =
        (static_cast<int64_t>(head) * padded_query_rows + row) * head_dim + dim;
    int64_t const stats_index = static_cast<int64_t>(head) * padded_query_rows + row;
    float const history_lse = history_log_sum_exp[stats_index];
    float const current_lse = current_log_sum_exp[stats_index];
    bool const history_valid = isfinite(history_lse);
    bool const current_valid = isfinite(current_lse);
    float value = 0.0F;
    if (history_valid || current_valid) {
        float const global_lse = history_valid && current_valid
                                     ? fmaxf(history_lse, current_lse)
                                     : (history_valid ? history_lse : current_lse);
        float const history_weight = history_valid ? expf(history_lse - global_lse) : 0.0F;
        float const current_weight = current_valid ? expf(current_lse - global_lse) : 0.0F;
        float const denominator = history_weight + current_weight;
        value = (history_weight * __bfloat162float(history_context[source_index]) +
                 current_weight * __bfloat162float(current_context[source_index])) /
                denominator;
    }
    destination[element] = __float2bfloat16_rn(value);
}

__device__ float warp_sum(float value) {
    for (int32_t offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

__device__ float warp_max(float value) {
    for (int32_t offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

__device__ float block_sum(float value, float* warp_scratch) {
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / 32;
    value = warp_sum(value);
    if (lane == 0) {
        warp_scratch[warp] = value;
    }
    __syncthreads();
    value = warp == 0 && lane < kAttentionWarps ? warp_scratch[lane] : 0.0F;
    if (warp == 0) {
        value = warp_sum(value);
    }
    if (threadIdx.x == 0) {
        warp_scratch[0] = value;
    }
    __syncthreads();
    float const result = warp_scratch[0];
    // Every thread must consume the shared result before the next reduction
    // lets lane zero of each warp reuse the same scratch slots.
    __syncthreads();
    return result;
}

__device__ float block_max(float value, float* warp_scratch) {
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / 32;
    value = warp_max(value);
    if (lane == 0) {
        warp_scratch[warp] = value;
    }
    __syncthreads();
    value = warp == 0 && lane < kAttentionWarps ? warp_scratch[lane] : -INFINITY;
    if (warp == 0) {
        value = warp_max(value);
    }
    if (threadIdx.x == 0) {
        warp_scratch[0] = value;
    }
    __syncthreads();
    float const result = warp_scratch[0];
    __syncthreads();
    return result;
}

std::size_t fused_decode_shared_memory_bytes(int32_t num_query_heads, int32_t num_kv_heads,
                                             int32_t head_dim) noexcept {
    if (num_query_heads <= 0 || num_kv_heads <= 0 || num_query_heads % num_kv_heads != 0 ||
        head_dim <= 0) {
        return 0;
    }
    auto const query_group_size = static_cast<std::uint64_t>(num_query_heads / num_kv_heads);
    auto const group_elements = query_group_size * static_cast<std::uint64_t>(head_dim);
    // Query + FP32 output accumulator, tile scores, six per-query-head
    // normalization/merge scalars, and one value per warp for reductions.
    auto const floats = 2U * group_elements +
                        query_group_size * static_cast<std::uint64_t>(kFusedDecodeTileRows) +
                        6U * query_group_size + static_cast<std::uint64_t>(kAttentionWarps);
    if (floats > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return 0;
    }
    return static_cast<std::size_t>(floats) * sizeof(float);
}

bool fused_decode_shape_supported(int32_t history_rows, int32_t num_query_heads,
                                  int32_t num_kv_heads, int32_t head_dim) noexcept {
    auto const shared_bytes =
        fused_decode_shared_memory_bytes(num_query_heads, num_kv_heads, head_dim);
    return history_rows > 0 && history_rows <= kFusedDecodeHistoryRowsLimit && shared_bytes > 0 &&
           shared_bytes <= kFusedDecodeSharedMemoryLimit;
}

// GQA-aware direct Sq=1 attention for the latency-sensitive history buckets.
// One block owns one KV head and all query heads in its group, so history K/V
// are shared across the group rather than reread by one block per query head.
// The online tiled softmax is O((H+1)*Hq*D), reads caller-owned cache rows in
// place, and writes only the exact one-token context.
__global__ void fused_single_token_attention(
    __nv_bfloat16 const* history_k, __nv_bfloat16 const* history_v, __nv_bfloat16 const* query,
    __nv_bfloat16 const* current_k, __nv_bfloat16 const* current_v,
    int32_t const* history_length_ptr, __nv_bfloat16* destination, int32_t history_rows,
    int32_t num_query_heads, int32_t num_kv_heads, int32_t head_dim, int32_t chunk_limit) {
    extern __shared__ float shared[];
    int32_t const query_group_size = num_query_heads / num_kv_heads;
    int32_t const group_elements = query_group_size * head_dim;
    float* shared_query = shared;
    float* scores = shared_query + group_elements;
    float* accumulator = scores + query_group_size * kFusedDecodeTileRows;
    float* running_max = accumulator + group_elements;
    float* running_sum = running_max + query_group_size;
    float* tile_max = running_sum + query_group_size;
    float* tile_sum = tile_max + query_group_size;
    float* old_scale = tile_sum + query_group_size;
    float* tile_scale = old_scale + query_group_size;
    float* reduction = tile_scale + query_group_size;
    __shared__ int32_t history_length;
    __shared__ int32_t request_valid;

    int32_t const kv_head = static_cast<int32_t>(blockIdx.x);
    int32_t const thread = static_cast<int32_t>(threadIdx.x);
    int32_t const warp = thread / 32;
    int32_t const lane = thread & 31;
    if (thread == 0) {
        history_length = *history_length_ptr;
        request_valid = valid_request(history_length, history_rows, 1, chunk_limit) ? 1 : 0;
    }
    __syncthreads();

    if (request_valid == 0) {
        for (int32_t element = thread; element < group_elements; element += kAttentionThreads) {
            int32_t const local_query_head = element / head_dim;
            int32_t const dim = element % head_dim;
            int32_t const query_head = kv_head * query_group_size + local_query_head;
            destination[static_cast<int64_t>(query_head) * head_dim + dim] =
                __float2bfloat16_rn(0.0F);
        }
        return;
    }

    for (int32_t element = thread; element < group_elements; element += kAttentionThreads) {
        int32_t const local_query_head = element / head_dim;
        int32_t const dim = element % head_dim;
        int32_t const query_head = kv_head * query_group_size + local_query_head;
        shared_query[element] =
            __bfloat162float(query[static_cast<int64_t>(query_head) * head_dim + dim]);
        accumulator[element] = 0.0F;
    }
    if (thread < query_group_size) {
        running_max[thread] = -INFINITY;
        running_sum[thread] = 0.0F;
    }
    __syncthreads();

    int32_t const total_rows = history_length + 1;
    float const attention_scale = rsqrtf(static_cast<float>(head_dim));
    for (int32_t tile_begin = 0; tile_begin < total_rows; tile_begin += kFusedDecodeTileRows) {
        int32_t const tile_rows = min(kFusedDecodeTileRows, total_rows - tile_begin);

        // Each warp owns complete dot products. Tasks are interleaved across
        // query heads and rows, and all lanes participate even when D < 32.
        for (int32_t task = warp; task < query_group_size * kFusedDecodeTileRows;
             task += kAttentionWarps) {
            int32_t const local_query_head = task / kFusedDecodeTileRows;
            int32_t const tile_row = task % kFusedDecodeTileRows;
            if (tile_row >= tile_rows) {
                continue;
            }
            int32_t const logical_row = tile_begin + tile_row;
            bool const is_current = logical_row == history_length;
            float product = 0.0F;
            for (int32_t dim = lane; dim < head_dim; dim += 32) {
                int64_t const key_index =
                    is_current
                        ? static_cast<int64_t>(kv_head) * head_dim + dim
                        : (static_cast<int64_t>(logical_row) * num_kv_heads + kv_head) * head_dim +
                              dim;
                auto const* key = is_current ? current_k : history_k;
                product += shared_query[local_query_head * head_dim + dim] *
                           __bfloat162float(key[key_index]);
            }
            product = warp_sum(product);
            if (lane == 0) {
                scores[local_query_head * kFusedDecodeTileRows + tile_row] =
                    product * attention_scale;
            }
        }
        __syncthreads();

        for (int32_t local_query_head = 0; local_query_head < query_group_size;
             ++local_query_head) {
            float maximum = -INFINITY;
            for (int32_t row = thread; row < tile_rows; row += kAttentionThreads) {
                maximum = fmaxf(maximum, scores[local_query_head * kFusedDecodeTileRows + row]);
            }
            maximum = block_max(maximum, reduction);
            float sum = 0.0F;
            for (int32_t row = thread; row < tile_rows; row += kAttentionThreads) {
                auto& score = scores[local_query_head * kFusedDecodeTileRows + row];
                score = expf(score - maximum);
                sum += score;
            }
            sum = block_sum(sum, reduction);
            if (thread == 0) {
                tile_max[local_query_head] = maximum;
                tile_sum[local_query_head] = sum;
            }
        }
        __syncthreads();

        if (thread < query_group_size) {
            bool const has_previous = isfinite(running_max[thread]);
            float const merged_max =
                has_previous ? fmaxf(running_max[thread], tile_max[thread]) : tile_max[thread];
            old_scale[thread] = has_previous ? expf(running_max[thread] - merged_max) : 0.0F;
            tile_scale[thread] = expf(tile_max[thread] - merged_max);
            running_sum[thread] =
                running_sum[thread] * old_scale[thread] + tile_sum[thread] * tile_scale[thread];
            running_max[thread] = merged_max;
        }
        __syncthreads();

        for (int32_t element = thread; element < group_elements; element += kAttentionThreads) {
            int32_t const local_query_head = element / head_dim;
            int32_t const dim = element % head_dim;
            float tile_value = 0.0F;
            for (int32_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
                int32_t const logical_row = tile_begin + tile_row;
                bool const is_current = logical_row == history_length;
                int64_t const value_index =
                    is_current
                        ? static_cast<int64_t>(kv_head) * head_dim + dim
                        : (static_cast<int64_t>(logical_row) * num_kv_heads + kv_head) * head_dim +
                              dim;
                auto const* value = is_current ? current_v : history_v;
                tile_value += scores[local_query_head * kFusedDecodeTileRows + tile_row] *
                              __bfloat162float(value[value_index]);
            }
            accumulator[element] = accumulator[element] * old_scale[local_query_head] +
                                   tile_value * tile_scale[local_query_head];
        }
        __syncthreads();
    }

    for (int32_t element = thread; element < group_elements; element += kAttentionThreads) {
        int32_t const local_query_head = element / head_dim;
        int32_t const dim = element % head_dim;
        int32_t const query_head = kv_head * query_group_size + local_query_head;
        destination[static_cast<int64_t>(query_head) * head_dim + dim] =
            __float2bfloat16_rn(accumulator[element] / running_sum[local_query_head]);
    }
}

bool launch_fused_single_token_attention(
    __nv_bfloat16 const* history_k, __nv_bfloat16 const* history_v, __nv_bfloat16 const* query,
    __nv_bfloat16 const* current_k, __nv_bfloat16 const* current_v, int32_t const* history_length,
    __nv_bfloat16* destination, int32_t history_rows, int32_t num_query_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t chunk_limit, cudaStream_t stream) noexcept {
    auto const shared_bytes =
        fused_decode_shared_memory_bytes(num_query_heads, num_kv_heads, head_dim);
    if (history_k == nullptr || history_v == nullptr || query == nullptr || current_k == nullptr ||
        current_v == nullptr || history_length == nullptr || destination == nullptr ||
        !fused_decode_shape_supported(history_rows, num_query_heads, num_kv_heads, head_dim) ||
        chunk_limit <= 0) {
        return false;
    }
    fused_single_token_attention<<<num_kv_heads, kAttentionThreads, shared_bytes, stream>>>(
        history_k, history_v, query, current_k, current_v, history_length, destination,
        history_rows, num_query_heads, num_kv_heads, head_dim, chunk_limit);
    return cudaPeekAtLastError() == cudaSuccess;
}

__global__ void
single_token_segmented_context(__nv_bfloat16 const* query, __nv_bfloat16 const* current_k,
                               __nv_bfloat16 const* current_v, __nv_bfloat16 const* history_context,
                               float const* history_log_sum_exp, int32_t const* request_valid,
                               __nv_bfloat16* destination, int32_t num_query_heads,
                               int32_t num_kv_heads, int32_t head_dim, bool has_history) {
    __shared__ float partial[kAttentionThreads];
    int32_t const query_head = static_cast<int32_t>(blockIdx.x);
    if (query_head >= num_query_heads) {
        return;
    }
    int32_t const group_size = num_query_heads / num_kv_heads;
    int32_t const kv_head = query_head / group_size;
    if (*request_valid == 0) {
        if (threadIdx.x < head_dim) {
            destination[query_head * head_dim + threadIdx.x] = __float2bfloat16_rn(0.0F);
        }
        return;
    }
    if (!has_history) {
        if (threadIdx.x < head_dim) {
            destination[query_head * head_dim + threadIdx.x] =
                current_v[kv_head * head_dim + threadIdx.x];
        }
        return;
    }

    float product = 0.0F;
    if (threadIdx.x < head_dim) {
        product = __bfloat162float(query[query_head * head_dim + threadIdx.x]) *
                  __bfloat162float(current_k[kv_head * head_dim + threadIdx.x]);
    }
    partial[threadIdx.x] = product;
    __syncthreads();
    for (int32_t stride = kAttentionThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x < head_dim) {
        float const history_lse = history_log_sum_exp[query_head];
        float const current_lse = partial[0] / sqrtf(static_cast<float>(head_dim));
        bool const history_valid = isfinite(history_lse);
        float const global_lse = history_valid ? fmaxf(history_lse, current_lse) : current_lse;
        float const history_weight = history_valid ? expf(history_lse - global_lse) : 0.0F;
        float const current_weight = expf(current_lse - global_lse);
        float const denominator = history_weight + current_weight;
        float const history_value =
            __bfloat162float(history_context[query_head * head_dim + threadIdx.x]);
        float const current_value = __bfloat162float(current_v[kv_head * head_dim + threadIdx.x]);
        destination[query_head * head_dim + threadIdx.x] = __float2bfloat16_rn(
            (history_weight * history_value + current_weight * current_value) / denominator);
    }
}

__global__ void copy_padded_context(__nv_bfloat16 const* source, __nv_bfloat16* destination,
                                    int32_t query_rows, int32_t padded_query_rows,
                                    int32_t num_query_heads, int32_t head_dim) {
    int64_t const element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const count = static_cast<int64_t>(num_query_heads) * query_rows * head_dim;
    if (element >= count) {
        return;
    }
    int32_t const dim = element % head_dim;
    int64_t const row_and_head = element / head_dim;
    int32_t const row = row_and_head % query_rows;
    int32_t const head = row_and_head / query_rows;
    int64_t const source_index =
        (static_cast<int64_t>(head) * padded_query_rows + row) * head_dim + dim;
    destination[element] = source[source_index];
}

bool is_linear(nvinfer1::DynamicPluginTensorDesc const& desc) noexcept {
    return desc.desc.format == nvinfer1::TensorFormat::kLINEAR;
}

bool valid_fields(int32_t abi_version, int32_t num_query_heads, int32_t num_kv_heads,
                  int32_t head_dim, int32_t chunk_limit) noexcept {
    return abi_version == kNativeContiguousAttentionPluginAbi && num_query_heads > 0 &&
           num_kv_heads > 0 && num_query_heads % num_kv_heads == 0 && head_dim > 0 &&
           head_dim <= kAttentionThreads && chunk_limit > 0;
}

bool prepare_for_shape(CudnnAttentionExecutor* attention, int32_t history_rows,
                       int32_t padded_query_rows) noexcept {
    if (attention == nullptr) {
        return false;
    }
    bool const cold = history_rows == 1;
    bool const single_token = padded_query_rows == 1;
    if (cold) {
        // Sq=1 is fully covered by the CUDA single-current-token path.
        return single_token || attention->prepare_current(padded_query_rows);
    }
    return single_token ? attention->prepare_history(history_rows, padded_query_rows)
                        : attention->prepare(history_rows, padded_query_rows);
}

bool valid_runtime_shapes(nvinfer1::PluginTensorDesc const* inputs, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* outputs, int32_t nb_outputs,
                          int32_t num_query_heads, int32_t num_kv_heads, int32_t head_dim,
                          int32_t chunk_limit) noexcept {
    if (inputs == nullptr || outputs == nullptr ||
        nb_inputs != kNativeContiguousAttentionInputCount ||
        nb_outputs != kNativeContiguousAttentionOutputCount) {
        return false;
    }
    for (int32_t index = 0; index < 2; ++index) {
        if (inputs[index].dims.nbDims != 2 ||
            inputs[index].format != nvinfer1::TensorFormat::kLINEAR ||
            inputs[index].type != nvinfer1::DataType::kBF16) {
            return false;
        }
    }
    for (int32_t index = 2; index < 5; ++index) {
        if (inputs[index].dims.nbDims != 4 ||
            inputs[index].format != nvinfer1::TensorFormat::kLINEAR ||
            inputs[index].type != nvinfer1::DataType::kBF16 || inputs[index].dims.d[0] != 1) {
            return false;
        }
    }
    if (inputs[5].dims.nbDims != 1 || inputs[5].dims.d[0] != 1 ||
        inputs[5].type != nvinfer1::DataType::kINT32) {
        return false;
    }

    int32_t const history_rows = inputs[0].dims.d[0];
    int32_t const query_rows = inputs[2].dims.d[2];
    if (history_rows <= 0 || query_rows <= 0 || query_rows > chunk_limit ||
        inputs[0].dims.d[1] != num_kv_heads * head_dim || inputs[1].dims.d[0] != history_rows ||
        inputs[1].dims.d[1] != num_kv_heads * head_dim || inputs[2].dims.d[1] != num_query_heads ||
        inputs[2].dims.d[3] != head_dim) {
        return false;
    }
    for (int32_t index = 3; index < 5; ++index) {
        if (inputs[index].dims.d[1] != num_kv_heads || inputs[index].dims.d[2] != query_rows ||
            inputs[index].dims.d[3] != head_dim) {
            return false;
        }
    }
    if (outputs[0].type != nvinfer1::DataType::kBF16 ||
        outputs[0].format != nvinfer1::TensorFormat::kLINEAR || outputs[0].dims.nbDims != 4 ||
        outputs[0].dims.d[0] != 1 || outputs[0].dims.d[1] != num_query_heads ||
        outputs[0].dims.d[2] != query_rows || outputs[0].dims.d[3] != head_dim) {
        return false;
    }
    return true;
}

} // namespace

bool launch_segmented_context_merge_for_testing(void const* history_context,
                                                void const* current_context,
                                                float const* history_log_sum_exp,
                                                float const* current_log_sum_exp, void* destination,
                                                int32_t query_rows, int32_t padded_query_rows,
                                                int32_t num_query_heads, int32_t head_dim,
                                                cudaStream_t stream) noexcept {
    if (history_context == nullptr || current_context == nullptr ||
        history_log_sum_exp == nullptr || current_log_sum_exp == nullptr ||
        destination == nullptr || query_rows <= 0 || padded_query_rows < query_rows ||
        num_query_heads <= 0 || head_dim <= 0) {
        return false;
    }
    auto const elements = static_cast<std::int64_t>(num_query_heads) * query_rows * head_dim;
    if (elements > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    int32_t const blocks =
        static_cast<int32_t>((elements + kAttentionThreads - 1) / kAttentionThreads);
    combine_segmented_context<<<blocks, kAttentionThreads, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(history_context),
        static_cast<__nv_bfloat16 const*>(current_context), history_log_sum_exp,
        current_log_sum_exp, static_cast<__nv_bfloat16*>(destination), query_rows,
        padded_query_rows, num_query_heads, head_dim);
    return cudaPeekAtLastError() == cudaSuccess;
}

NativeContiguousAttentionPlugin::NativeContiguousAttentionPlugin(int32_t abi_version,
                                                                 int32_t num_query_heads,
                                                                 int32_t num_kv_heads,
                                                                 int32_t head_dim,
                                                                 int32_t chunk_limit) noexcept
    : abi_version_(abi_version), num_query_heads_(num_query_heads), num_kv_heads_(num_kv_heads),
      head_dim_(head_dim), chunk_limit_(chunk_limit) {}

NativeContiguousAttentionPlugin::~NativeContiguousAttentionPlugin() = default;

nvinfer1::IPluginCapability* NativeContiguousAttentionPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
    switch (type) {
    case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuildV2*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

nvinfer1::IPluginV3* NativeContiguousAttentionPlugin::clone() noexcept {
    return new (std::nothrow) NativeContiguousAttentionPlugin(
        abi_version_, num_query_heads_, num_kv_heads_, head_dim_, chunk_limit_);
}

nvinfer1::AsciiChar const* NativeContiguousAttentionPlugin::getPluginName() const noexcept {
    return kNativeContiguousAttentionPluginName;
}

nvinfer1::AsciiChar const* NativeContiguousAttentionPlugin::getPluginVersion() const noexcept {
    return kNativeContiguousAttentionPluginVersion;
}

nvinfer1::AsciiChar const* NativeContiguousAttentionPlugin::getPluginNamespace() const noexcept {
    return "";
}

int32_t NativeContiguousAttentionPlugin::getNbOutputs() const noexcept {
    return kNativeContiguousAttentionOutputCount;
}

int32_t NativeContiguousAttentionPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nb_inputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nb_outputs) noexcept {
    if (!valid_fields(abi_version_, num_query_heads_, num_kv_heads_, head_dim_, chunk_limit_) ||
        inputs == nullptr || outputs == nullptr ||
        nb_inputs != kNativeContiguousAttentionInputCount ||
        nb_outputs != kNativeContiguousAttentionOutputCount) {
        return -1;
    }
    return 0;
}

int32_t NativeContiguousAttentionPlugin::getOutputDataTypes(nvinfer1::DataType* output_types,
                                                            int32_t nb_outputs,
                                                            nvinfer1::DataType const* input_types,
                                                            int32_t nb_inputs) const noexcept {
    if (output_types == nullptr || input_types == nullptr ||
        nb_inputs != kNativeContiguousAttentionInputCount ||
        nb_outputs != kNativeContiguousAttentionOutputCount) {
        return -1;
    }
    output_types[0] = input_types[2];
    return 0;
}

int32_t NativeContiguousAttentionPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t nb_inputs, nvinfer1::DimsExprs const*,
    int32_t nb_shape_inputs, nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
    nvinfer1::IExprBuilder&) noexcept {
    if (inputs == nullptr || outputs == nullptr ||
        nb_inputs != kNativeContiguousAttentionInputCount || nb_shape_inputs != 0 ||
        nb_outputs != kNativeContiguousAttentionOutputCount) {
        return -1;
    }
    outputs[0] = inputs[2];
    return 0;
}

bool NativeContiguousAttentionPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* in_out, int32_t nb_inputs,
    int32_t nb_outputs) noexcept {
    if (in_out == nullptr || nb_inputs != kNativeContiguousAttentionInputCount ||
        nb_outputs != kNativeContiguousAttentionOutputCount || pos < 0 ||
        pos >= nb_inputs + nb_outputs || !is_linear(in_out[pos])) {
        return false;
    }
    if (pos <= 4 || pos == kNativeContiguousAttentionInputCount) {
        return in_out[pos].desc.type == nvinfer1::DataType::kBF16;
    }
    return in_out[pos].desc.type == nvinfer1::DataType::kINT32;
}

size_t NativeContiguousAttentionPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nb_inputs,
    nvinfer1::DynamicPluginTensorDesc const*, int32_t) const noexcept {
    int32_t padded_query_rows = chunk_limit_;
    if (inputs != nullptr && nb_inputs == kNativeContiguousAttentionInputCount) {
        const int32_t profile_query_rows = inputs[2].max.d[2];
        if (profile_query_rows == 1) {
            padded_query_rows = 1;
        }
    }
    return cudnn_attention_workspace_size(
        CudnnAttentionConfig{
            num_query_heads_,
            num_kv_heads_,
            head_dim_,
            chunk_limit_,
        },
        padded_query_rows);
}

int32_t NativeContiguousAttentionPlugin::getAliasedInput(int32_t) noexcept {
    return -1;
}

nvinfer1::AsciiChar const* NativeContiguousAttentionPlugin::getMetadataString() noexcept {
    return "abi=2;history_layout=token_major_THD;"
           "current_layout=head_major_BHTD;dtype=bf16;gqa=true;"
           "causal=lower_right;workspace=bounded;"
           "cache=read_only_segmented;copies_history=false;"
           "normalization=log_sum_exp;"
           "cold_shape=T1;noncold_shape=Tge2;"
           "scalar_validation=H0_requires_T1_Hpositive_requires_Tge2;"
           "decode=fused_direct_gqa_Ple512_else_history_cudnn_plus_fused_current_merge;"
           "cold_decode=fused_current_no_cudnn;"
           "performance=hybrid_fused_decode_online_softmax_and_cudnn_sdpa_9_20";
}

int32_t NativeContiguousAttentionPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
                                                       int32_t nb_inputs,
                                                       nvinfer1::PluginTensorDesc const* outputs,
                                                       int32_t nb_outputs) noexcept {
    if (!valid_runtime_shapes(inputs, nb_inputs, outputs, nb_outputs, num_query_heads_,
                              num_kv_heads_, head_dim_, chunk_limit_)) {
        return -1;
    }
    const int32_t query_rows = inputs[2].dims.d[2];
    const int32_t padded_query_rows = query_rows == 1 ? 1 : chunk_limit_;
    if (query_rows == 1 && fused_decode_shape_supported(inputs[0].dims.d[0], num_query_heads_,
                                                        num_kv_heads_, head_dim_)) {
        return 0;
    }
    if (!native_cudnn_attention_available()) {
        return -1;
    }
    if (!attention_) {
        attention_ = make_cudnn_attention_executor(CudnnAttentionConfig{
            num_query_heads_,
            num_kv_heads_,
            head_dim_,
            chunk_limit_,
        });
    }
    // Runtime shape convention: T==1 is reserved for H==0. Any non-empty
    // history binds T>=2 (H==1 uses one padded row). This makes the cold path
    // host-visible without synchronously reading the device H scalar.
    bool const prepared =
        prepare_for_shape(attention_.get(), inputs[0].dims.d[0], padded_query_rows);
    return prepared ? 0 : -1;
}

int32_t NativeContiguousAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                                                 nvinfer1::PluginTensorDesc const* output_desc,
                                                 void const* const* inputs, void* const* outputs,
                                                 void* workspace, cudaStream_t stream) noexcept {
    if (inputs == nullptr || outputs == nullptr || outputs[0] == nullptr ||
        onShapeChange(input_desc, kNativeContiguousAttentionInputCount, output_desc,
                      kNativeContiguousAttentionOutputCount) != 0) {
        return -1;
    }
    for (int32_t index = 0; index < kNativeContiguousAttentionInputCount; ++index) {
        if (inputs[index] == nullptr) {
            return -1;
        }
    }

    int32_t const history_rows = input_desc[0].dims.d[0];
    int32_t const query_rows = input_desc[2].dims.d[2];
    int32_t const padded_query_rows = query_rows == 1 ? 1 : chunk_limit_;
    auto const* history_length = static_cast<int32_t const*>(inputs[5]);
    auto const* history_k = static_cast<__nv_bfloat16 const*>(inputs[0]);
    auto const* history_v = static_cast<__nv_bfloat16 const*>(inputs[1]);
    auto const* query = static_cast<__nv_bfloat16 const*>(inputs[2]);
    auto const* current_k = static_cast<__nv_bfloat16 const*>(inputs[3]);
    auto const* current_v = static_cast<__nv_bfloat16 const*>(inputs[4]);
    auto* context = static_cast<__nv_bfloat16*>(outputs[0]);
    if (query_rows == 1 &&
        fused_decode_shape_supported(history_rows, num_query_heads_, num_kv_heads_, head_dim_)) {
        return launch_fused_single_token_attention(
                   history_k, history_v, query, current_k, current_v, history_length, context,
                   history_rows, num_query_heads_, num_kv_heads_, head_dim_, chunk_limit_, stream)
                   ? 0
                   : -1;
    }
    const auto workspace_bytes = cudnn_attention_workspace_size(
        CudnnAttentionConfig{
            num_query_heads_,
            num_kv_heads_,
            head_dim_,
            chunk_limit_,
        },
        padded_query_rows);
    bool const cold = history_rows == 1;
    bool const prepared = prepare_for_shape(attention_.get(), history_rows, padded_query_rows);
    if (workspace == nullptr || workspace_bytes == 0 || !prepared) {
        return -1;
    }

    auto* workspace_bytes_ptr = static_cast<std::uint8_t*>(workspace);
    auto* sequence_length_q = reinterpret_cast<int32_t*>(workspace_bytes_ptr);
    auto* sequence_length_history = sequence_length_q + 1;
    auto* sequence_length_current = sequence_length_history + 1;
    auto* request_valid = sequence_length_current + 1;
    std::size_t offset =
        align_up(sizeof(int32_t) * kCudnnAttentionControlScalarCount, kWorkspaceAlignment);
    if (offset == 0) {
        return -1;
    }

    const auto padded_query_elements = static_cast<std::uint64_t>(num_query_heads_) *
                                       static_cast<std::uint64_t>(padded_query_rows) *
                                       static_cast<std::uint64_t>(head_dim_);
    const auto padded_kv_elements = static_cast<std::uint64_t>(num_kv_heads_) *
                                    static_cast<std::uint64_t>(padded_query_rows) *
                                    static_cast<std::uint64_t>(head_dim_);
    const auto padded_stats_elements = static_cast<std::uint64_t>(num_query_heads_) *
                                       static_cast<std::uint64_t>(padded_query_rows);
    if (padded_query_elements > std::numeric_limits<std::size_t>::max() / sizeof(__nv_bfloat16) ||
        padded_kv_elements > std::numeric_limits<std::size_t>::max() / sizeof(__nv_bfloat16) ||
        padded_stats_elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return -1;
    }

    auto reserve_region = [&](std::size_t bytes) -> std::uint8_t* {
        if (offset > workspace_bytes || bytes > workspace_bytes - offset) {
            return nullptr;
        }
        auto* result = workspace_bytes_ptr + offset;
        offset = align_up(offset + bytes, kWorkspaceAlignment);
        if (offset == 0 || offset > workspace_bytes) {
            return nullptr;
        }
        return result;
    };
    const auto padded_query_bytes =
        static_cast<std::size_t>(padded_query_elements) * sizeof(__nv_bfloat16);
    const auto padded_kv_bytes =
        static_cast<std::size_t>(padded_kv_elements) * sizeof(__nv_bfloat16);
    const auto padded_stats_bytes = static_cast<std::size_t>(padded_stats_elements) * sizeof(float);
    auto* padded_query = reinterpret_cast<__nv_bfloat16*>(reserve_region(padded_query_bytes));
    auto* padded_current_k = reinterpret_cast<__nv_bfloat16*>(reserve_region(padded_kv_bytes));
    auto* padded_current_v = reinterpret_cast<__nv_bfloat16*>(reserve_region(padded_kv_bytes));
    auto* history_context = reinterpret_cast<__nv_bfloat16*>(reserve_region(padded_query_bytes));
    auto* current_context = reinterpret_cast<__nv_bfloat16*>(reserve_region(padded_query_bytes));
    auto* history_log_sum_exp = reinterpret_cast<float*>(reserve_region(padded_stats_bytes));
    auto* current_log_sum_exp = reinterpret_cast<float*>(reserve_region(padded_stats_bytes));
    if (padded_query == nullptr || padded_current_k == nullptr || padded_current_v == nullptr ||
        history_context == nullptr || current_context == nullptr ||
        history_log_sum_exp == nullptr || current_log_sum_exp == nullptr) {
        return -1;
    }
    if (offset > workspace_bytes || workspace_bytes - offset < kCudnnAttentionPlanWorkspaceLimit) {
        return -1;
    }

    const auto query_elements =
        static_cast<std::int64_t>(num_query_heads_) * query_rows * head_dim_;
    const auto kv_elements = static_cast<std::int64_t>(num_kv_heads_) * query_rows * head_dim_;
    const auto query_blocks =
        static_cast<int32_t>((query_elements + kAttentionThreads - 1) / kAttentionThreads);
    const auto kv_blocks =
        static_cast<int32_t>((kv_elements + kAttentionThreads - 1) / kAttentionThreads);
    bool const single_token = query_rows == 1;
    if (!single_token) {
        pack_padded_head_major<<<query_blocks, kAttentionThreads, 0, stream>>>(
            query, padded_query, query_rows, padded_query_rows, num_query_heads_, head_dim_);
        pack_padded_head_major<<<kv_blocks, kAttentionThreads, 0, stream>>>(
            current_k, padded_current_k, query_rows, padded_query_rows, num_kv_heads_, head_dim_);
        pack_padded_head_major<<<kv_blocks, kAttentionThreads, 0, stream>>>(
            current_v, padded_current_v, query_rows, padded_query_rows, num_kv_heads_, head_dim_);
        if (cudaPeekAtLastError() != cudaSuccess) {
            return -1;
        }
    }

    prepare_sequence_lengths<<<1, 1, 0, stream>>>(
        history_length, sequence_length_q, sequence_length_history, sequence_length_current,
        request_valid, history_rows, query_rows, chunk_limit_);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return -1;
    }

    auto const* history_query = single_token ? query : padded_query;
    if (!cold && !attention_->execute_history(history_query, history_k, history_v, history_context,
                                              history_log_sum_exp, sequence_length_q,
                                              sequence_length_history, workspace_bytes_ptr + offset,
                                              workspace_bytes - offset, stream)) {
        return -1;
    }

    if (single_token) {
        single_token_segmented_context<<<num_query_heads_, kAttentionThreads, 0, stream>>>(
            query, current_k, current_v, history_context, history_log_sum_exp, request_valid,
            context, num_query_heads_, num_kv_heads_, head_dim_, !cold);
        if (cudaPeekAtLastError() != cudaSuccess) {
            return -1;
        }
    } else if (!attention_->execute_current(padded_query, padded_current_k, padded_current_v,
                                            current_context, current_log_sum_exp, sequence_length_q,
                                            sequence_length_current, workspace_bytes_ptr + offset,
                                            workspace_bytes - offset, stream)) {
        return -1;
    }

    if (!single_token) {
        if (cold) {
            copy_padded_context<<<query_blocks, kAttentionThreads, 0, stream>>>(
                current_context, context, query_rows, padded_query_rows, num_query_heads_,
                head_dim_);
        } else if (!launch_segmented_context_merge_for_testing(
                       history_context, current_context, history_log_sum_exp, current_log_sum_exp,
                       context, query_rows, padded_query_rows, num_query_heads_, head_dim_,
                       stream)) {
            return -1;
        }
        zero_context_if_invalid<<<query_blocks, kAttentionThreads, 0, stream>>>(
            request_valid, context, query_elements);
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

nvinfer1::IPluginV3*
NativeContiguousAttentionPlugin::attachToContext(nvinfer1::IPluginResourceContext*) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const*
NativeContiguousAttentionPlugin::getFieldsToSerialize() noexcept {
    serialized_fields_[0] = {"abi_version", &abi_version_, nvinfer1::PluginFieldType::kINT32, 1};
    serialized_fields_[1] = {"num_query_heads", &num_query_heads_,
                             nvinfer1::PluginFieldType::kINT32, 1};
    serialized_fields_[2] = {"num_kv_heads", &num_kv_heads_, nvinfer1::PluginFieldType::kINT32, 1};
    serialized_fields_[3] = {"head_dim", &head_dim_, nvinfer1::PluginFieldType::kINT32, 1};
    serialized_fields_[4] = {"chunk_limit", &chunk_limit_, nvinfer1::PluginFieldType::kINT32, 1};
    serialized_fields_collection_.nbFields = static_cast<int32_t>(serialized_fields_.size());
    serialized_fields_collection_.fields = serialized_fields_.data();
    return &serialized_fields_collection_;
}

} // namespace trtmc::runtime_kv
