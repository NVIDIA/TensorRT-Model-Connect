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

__global__ void prepare_query(const __nv_bfloat16* query, float* scaled_query, float factor) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 128 * 128)
        scaled_query[index] = __fmul_rn(__bfloat162float(query[index]), factor);
}

__global__ void key_mask(const int32_t* indices, const int32_t* count, const int32_t* sizes,
                         uint8_t* mask, int tiles) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int rank = index / 128;
    if (rank >= tiles || rank >= *count)
        return;
    const int tile = indices[rank];
    assert(tile >= 0 && tile < tiles && sizes[tile] >= 0 && sizes[tile] <= 128);
    mask[tile * 128 + index % 128] = index % 128 < sizes[tile];
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

__global__ void softmax_short(float* scores, const uint8_t* mask, int keys) {
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

__global__ void softmax_regular(float* scores, const uint8_t* mask, int keys) {
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
                           int tile) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 128 * 128)
        output[index] = __float2bfloat16_rn(index / 128 < sizes[tile] ? fine[index] : 0.0F);
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
    // Reuse one head's arena and one complete query tile. There is no hidden
    // allocation, stream synchronization or full-sequence score matrix.
    for (int head = 0; head < heads; ++head) {
        const int64_t base = head * head_elements;
        prepare_keys<<<(head_elements + 255) / 256, 256, 0, stream>>>(
            key + base, value + base, scaled_key, float_value, head_elements, factor);
        require_cuda(cudaGetLastError());
        for (int tile = 0; tile < tiles; ++tile) {
            const int64_t map_row = static_cast<int64_t>(head) * tiles + tile;
            prepare_query<<<64, 256, 0, stream>>>(query + base + tile * 16384, scaled_query,
                                                  factor);
            require_cuda(cudaGetLastError());
            require_cuda(cudaMemsetAsync(mask, 0, rows, stream));
            key_mask<<<(rows + 255) / 256, 256, 0, stream>>>(indices + map_row * tiles,
                                                             counts + map_row, sizes, mask, tiles);
            require_cuda(cudaGetLastError());
            require_cublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, 128, 128, &one,
                                       scaled_key, 128, scaled_query, 128, &zero, scores, rows));
            if (rows <= 2048)
                softmax_short<<<128, 32, 0, stream>>>(scores, mask, rows);
            else
                softmax_regular<<<128, 1024, 0, stream>>>(scores, mask, rows);
            require_cuda(cudaGetLastError());
            require_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 128, 128, rows, &one,
                                       float_value, 128, scores, rows, &zero, fine, 128));
            store_fine<<<64, 256, 0, stream>>>(fine, output + base + tile * 16384, sizes, tile);
            require_cuda(cudaGetLastError());
        }
    }
}

} // namespace trtmc::minimax_h3::fp32_vsa
