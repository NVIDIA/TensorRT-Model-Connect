/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if TRTMC_HAS_TRT

#include "plugins/top4_logits_refinement_plugin.h"

#include <cfloat>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace trtmc {
namespace {

constexpr int32_t kThreads = 256;
constexpr int32_t kBlocks = 32;
constexpr int32_t kCandidates = 4;

struct Top4Partial {
    float values[kCandidates];
    int32_t indices[kCandidates];
};

__device__ void initialize_candidates(float* values, int32_t* indices, int32_t vocab_size) {
    for (int32_t candidate = 0; candidate < kCandidates; ++candidate) {
        values[candidate] = -FLT_MAX;
        indices[candidate] = vocab_size;
    }
}

__device__ void insert_candidate(float value, int32_t index, float* values, int32_t* indices) {
    for (int32_t candidate = 0; candidate < kCandidates; ++candidate) {
        if (index == indices[candidate])
            return;
        const bool better =
            value > values[candidate] || (value == values[candidate] && index < indices[candidate]);
        if (!better)
            continue;
        for (int32_t shifted = kCandidates - 1; shifted > candidate; --shifted) {
            values[shifted] = values[shifted - 1];
            indices[shifted] = indices[shifted - 1];
        }
        values[candidate] = value;
        indices[candidate] = index;
        return;
    }
}

__global__ void find_top4_and_copy_kernel(float const* logits, float* output,
                                          Top4Partial* block_partials, int32_t vocab_size) {
    __shared__ Top4Partial thread_partials[kThreads];
    Top4Partial local;
    initialize_candidates(local.values, local.indices, vocab_size);

    const int32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t global_stride = blockDim.x * gridDim.x;
    for (int32_t index = global_thread; index < vocab_size; index += global_stride) {
        const float value = logits[index];
        output[index] = value;
        insert_candidate(value, index, local.values, local.indices);
    }
    thread_partials[threadIdx.x] = local;
    __syncthreads();

    if (threadIdx.x == 0) {
        Top4Partial block;
        initialize_candidates(block.values, block.indices, vocab_size);
        for (int32_t thread = 0; thread < blockDim.x; ++thread) {
            const Top4Partial partial = thread_partials[thread];
            for (int32_t candidate = 0; candidate < kCandidates; ++candidate)
                insert_candidate(partial.values[candidate], partial.indices[candidate],
                                 block.values, block.indices);
        }
        block_partials[blockIdx.x] = block;
    }
}

__global__ void refine_top4_kernel(__half const* hidden, float const* weights, float const* bias,
                                   Top4Partial const* block_partials, float* output,
                                   int32_t hidden_size, int32_t vocab_size) {
    __shared__ float partial_sums[kCandidates][kThreads];
    __shared__ int32_t selected_indices[kCandidates];

    if (threadIdx.x == 0) {
        float values[kCandidates];
        int32_t indices[kCandidates];
        initialize_candidates(values, indices, vocab_size);
        for (int32_t block = 0; block < kBlocks; ++block) {
            const Top4Partial partial = block_partials[block];
            for (int32_t candidate = 0; candidate < kCandidates; ++candidate)
                insert_candidate(partial.values[candidate], partial.indices[candidate], values,
                                 indices);
        }
        for (int32_t candidate = 0; candidate < kCandidates; ++candidate)
            selected_indices[candidate] = indices[candidate];
    }
    __syncthreads();

    float sums[kCandidates]{};
    for (int32_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const float activation = __half2float(hidden[column]);
        for (int32_t candidate = 0; candidate < kCandidates; ++candidate) {
            const int32_t index = selected_indices[candidate];
            sums[candidate] =
                fmaf(activation, weights[index * hidden_size + column], sums[candidate]);
        }
    }
    for (int32_t candidate = 0; candidate < kCandidates; ++candidate)
        partial_sums[candidate][threadIdx.x] = sums[candidate];
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            for (int32_t candidate = 0; candidate < kCandidates; ++candidate)
                partial_sums[candidate][threadIdx.x] +=
                    partial_sums[candidate][threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int32_t candidate = 0; candidate < kCandidates; ++candidate) {
            const int32_t index = selected_indices[candidate];
            output[index] = partial_sums[candidate][0] + bias[index];
        }
    }
}

} // namespace

int32_t launch_top4_logits_refinement(void const* logits, void const* hidden, void const* weights,
                                      void const* bias, void* output, void* workspace,
                                      int32_t hidden_size, int32_t vocab_size,
                                      cudaStream_t stream) noexcept {
    auto* block_partials = static_cast<Top4Partial*>(workspace);
    find_top4_and_copy_kernel<<<kBlocks, kThreads, 0, stream>>>(
        static_cast<float const*>(logits), static_cast<float*>(output), block_partials, vocab_size);
    refine_top4_kernel<<<1, kThreads, 0, stream>>>(
        static_cast<__half const*>(hidden), static_cast<float const*>(weights),
        static_cast<float const*>(bias), block_partials, static_cast<float*>(output), hidden_size,
        vocab_size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

size_t top4_logits_refinement_workspace_size() noexcept {
    return kBlocks * sizeof(Top4Partial);
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT
