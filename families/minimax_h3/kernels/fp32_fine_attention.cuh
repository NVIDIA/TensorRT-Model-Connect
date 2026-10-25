/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "vsa_workspace.h"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace trtmc::minimax_h3::fp32_vsa {

inline void require_cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("VSA FP32 fine attention: ") +
                                 cudaGetErrorString(status));
}

inline void require_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("VSA FP32 fine GEMM failed: " + std::to_string(status));
}

__global__ void prepare_keys(const __nv_bfloat16* key, const __nv_bfloat16* value,
                             float* scaled_key, float* float_value, int64_t count, float factor) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        scaled_key[index] = __fmul_rn(__bfloat162float(key[index]), factor);
        float_value[index] = __bfloat162float(value[index]);
    }
}

__global__ void prepare_query(const __nv_bfloat16* query, float* scaled_query, float factor,
                              int query_tiles = 1) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < query_tiles * 128 * 128)
        scaled_query[index] = __fmul_rn(__bfloat162float(query[index]), factor);
}

__global__ void key_mask(const int32_t* indices, const int32_t* count, const int32_t* sizes,
                         uint8_t* mask, int tiles, int query_tiles = 1) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = tiles * 128;
    const int query = index / rows;
    const int local = index % rows;
    const int rank = local / 128;
    if (query >= query_tiles || rank >= count[query])
        return;
    const int tile = indices[static_cast<int64_t>(query) * tiles + rank];
    assert(tile >= 0 && tile < tiles && sizes[tile] >= 0 && sizes[tile] <= 128);
    mask[static_cast<int64_t>(query) * rows + tile * 128 + local % 128] =
        local % 128 < sizes[tile];
}

template <bool Maximum>
__device__ float block_reduce(float value, float* shared) {
    for (int step = 16; step != 0; step /= 2) {
        const float other = __shfl_down_sync(0xffffffffU, value, step);
        value = Maximum ? fmaxf(value, other) : __fadd_rn(value, other);
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0)
        shared[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : (Maximum ? -FLT_MAX : 0.0F);
        for (int step = 16; step != 0; step /= 2) {
            const float other = __shfl_down_sync(0xffffffffU, value, step);
            value = Maximum ? fmaxf(value, other) : __fadd_rn(value, other);
        }
        if (threadIdx.x == 0)
            shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

template <typename Mask>
__global__ void softmax_short(float* scores, const Mask* mask, int keys, int mask_stride = 0) {
    mask += static_cast<int64_t>(blockIdx.x / 128) * mask_stride;
    float* row = scores + static_cast<int64_t>(blockIdx.x) * keys;
    float maximum = -FLT_MAX;
    for (int column = threadIdx.x; column < keys; column += 32)
        if (mask[column])
            maximum = fmaxf(maximum, row[column]);
    for (int step = 16; step != 0; step /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, step));
    float total = 0.0F;
    for (int column = threadIdx.x; column < keys; column += 32)
        if (mask[column])
            total = __fadd_rn(total, expf(__fsub_rn(row[column], maximum)));
    for (int step = 16; step != 0; step /= 2)
        total = __fadd_rn(total, __shfl_xor_sync(0xffffffffU, total, step));
    for (int column = threadIdx.x; column < keys; column += 32)
        row[column] = mask[column] ? __fdiv_rn(expf(__fsub_rn(row[column], maximum)), total) : 0.0F;
}

template <typename Mask>
__global__ void softmax_regular(float* scores, const Mask* mask, int keys, int mask_stride = 0) {
    mask += static_cast<int64_t>(blockIdx.x / 128) * mask_stride;
    __shared__ float reduction[32];
    float* row = scores + static_cast<int64_t>(blockIdx.x) * keys;
    const int full = keys <= 9216 ? 0 : keys / 4096 * 4096;
    float maximum = -FLT_MAX;
    for (int first = 4 * threadIdx.x; first < full; first += 4096) {
#pragma unroll
        for (int item = 0; item < 4; ++item)
            if (mask[first + item])
                maximum = fmaxf(maximum, row[first + item]);
    }
    for (int column = full + threadIdx.x; column < keys; column += blockDim.x)
        if (mask[column])
            maximum = fmaxf(maximum, row[column]);
    maximum = block_reduce<true>(maximum, reduction);
    float total = 0.0F;
    for (int first = 4 * threadIdx.x; first < full; first += 4096) {
#pragma unroll
        for (int item = 0; item < 4; ++item)
            if (mask[first + item])
                total = __fadd_rn(total, expf(__fsub_rn(row[first + item], maximum)));
    }
    for (int column = full + threadIdx.x; column < keys; column += blockDim.x)
        if (mask[column])
            total = __fadd_rn(total, expf(__fsub_rn(row[column], maximum)));
    total = block_reduce<false>(total, reduction);
    for (int column = threadIdx.x; column < keys; column += blockDim.x)
        row[column] = mask[column] ? __fdiv_rn(expf(__fsub_rn(row[column], maximum)), total) : 0.0F;
}

__global__ void store_fine(const float* fine, __nv_bfloat16* output, const int32_t* sizes,
                           int tile, int query_tiles = 1) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < query_tiles * 128 * 128)
        output[index] = __float2bfloat16_rn(
            index % (128 * 128) / 128 < sizes[tile + index / (128 * 128)] ? fine[index] : 0.0F);
}

__global__ void prepare_qk_batch(const float* key, const float* query, float* scores,
                                 const float** inputs, float** outputs, int keys, int queries) {
    const int index = threadIdx.x;
    if (index < queries) {
        inputs[index] = key;
        inputs[queries + index] = query + index * 128 * 128;
        outputs[index] = scores + static_cast<int64_t>(index) * 128 * keys;
    }
}

__global__ void prepare_pv_batch(const float* value, const float* probability, float* first,
                                 float* fine, const float** inputs, float** outputs,
                                 int keys, int queries) {
    const int products = queries > 1 ? queries : 2;
    const int index = threadIdx.x;
    if (index < products) {
        inputs[index] = value;
        inputs[products + index] =
            probability + static_cast<int64_t>(queries > 1 ? index : 0) * 128 * keys;
        outputs[index] = queries > 1 ? fine + index * 128 * 128 : (index == 0 ? first : fine);
    }
}

inline void run(const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
                const int32_t* indices, const int32_t* counts, const int32_t* sizes,
                __nv_bfloat16* output, uint8_t* workspace, const VsaWorkspaceLayout& layout,
                int heads, int tiles, cublasHandle_t handle, cudaStream_t stream) {
    const int rows = tiles * 128;
    const int64_t head_elements = static_cast<int64_t>(rows) * 128;
    auto* scaled_key = reinterpret_cast<float*>(workspace + layout.scaled_key_offset);
    auto* float_value = reinterpret_cast<float*>(workspace + layout.value_offset);
    auto* scaled_query = reinterpret_cast<float*>(workspace + layout.scaled_query_offset);
    auto* scores = reinterpret_cast<float*>(workspace + layout.score_offset);
    auto* fine = reinterpret_cast<float*>(workspace + layout.fine_output_offset);
    auto* mask = workspace + layout.key_mask_offset;
    const float factor = static_cast<float>(std::sqrt(1.0 / std::sqrt(128.0)));
    const float one = 1.0F, zero = 0.0F;
    require_cublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    require_cublas(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));
    // Reuse one head's K/V and fit independent query tiles into caller scratch.
    // There is no hidden allocation, synchronization or full-sequence score matrix.
    for (int head = 0; head < heads; ++head) {
        const int64_t base = head * head_elements;
        prepare_keys<<<(head_elements + 255) / 256, 256, 0, stream>>>(
            key + base, value + base, scaled_key, float_value, head_elements, factor);
        require_cuda(cudaGetLastError());
        for (int tile = 0; tile < tiles; tile += layout.query_tile_batch) {
            const int queries = std::min(layout.query_tile_batch, tiles - tile);
            const int query_elements = queries * 128 * 128;
            const int mask_elements = queries * rows;
            const int64_t map_row = static_cast<int64_t>(head) * tiles + tile;
            prepare_query<<<(query_elements + 255) / 256, 256, 0, stream>>>(
                query + base + tile * 16384, scaled_query, factor, queries);
            require_cuda(cudaGetLastError());
            require_cuda(cudaMemsetAsync(mask, 0, mask_elements, stream));
            key_mask<<<(mask_elements + 255) / 256, 256, 0, stream>>>(
                indices + map_row * tiles, counts + map_row, sizes, mask, tiles, queries);
            require_cuda(cudaGetLastError());
            if (queries == 1) {
                require_cublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, 128, 128, &one,
                                           scaled_key, 128, scaled_query, 128, &zero, scores, rows));
            } else {
                // PV output is dead during QK. Reuse its first bytes for the
                // device pointer table; QK never writes or reads PV data.
                auto** inputs = reinterpret_cast<const float**>(fine);
                auto** outputs = reinterpret_cast<float**>(
                    reinterpret_cast<uint8_t*>(fine) + 2 * queries * sizeof(const float*));
                prepare_qk_batch<<<1, 32, 0, stream>>>(scaled_key, scaled_query, scores,
                                                       inputs, outputs, rows, queries);
                require_cuda(cudaGetLastError());
                require_cublas(cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, 128, 128,
                                                  &one, inputs, 128, inputs + queries, 128, &zero,
                                                  outputs, rows, queries));
            }
            if (rows <= 2048)
                softmax_short<<<queries * 128, 32, 0, stream>>>(scores, mask, rows, rows);
            else
                softmax_regular<<<queries * 128, 1024, 0, stream>>>(scores, mask, rows, rows);
            require_cuda(cudaGetLastError());
            // Softmax no longer needs the mask. Its bytes hold the PV pointer
            // table. A single query retains the proven duplicate-PV reduction,
            // with dead Q and fine as disjoint destinations; larger batches
            // contain independent original query tiles with shared read-only V.
            const int products = queries > 1 ? queries : 2;
            auto** pv_inputs = reinterpret_cast<const float**>(mask);
            auto** pv_outputs = reinterpret_cast<float**>(mask + 2 * products * sizeof(const float*));
            prepare_pv_batch<<<1, 32, 0, stream>>>(float_value, scores, scaled_query, fine,
                                                   pv_inputs, pv_outputs, rows, queries);
            require_cuda(cudaGetLastError());
            require_cublas(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 128, 128, rows,
                                              &one, pv_inputs, 128, pv_inputs + products, rows, &zero,
                                              pv_outputs, 128, products));
            store_fine<<<(query_elements + 255) / 256, 256, 0, stream>>>(
                fine, output + base + tile * 16384, sizes, tile, queries);
            require_cuda(cudaGetLastError());
        }
    }
}

} // namespace trtmc::minimax_h3::fp32_vsa
