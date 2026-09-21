/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bark/runtime/sparse_multinomial_kernel.h"

#include <cfloat>
#include <curand_kernel.h>
#include <limits>

namespace trtmc {

namespace {

constexpr int kSamplerBlockSize = 128;

__device__ float torch_exponential_from_uniform(float value) {
    const float log_value = value >= 1.0F - FLT_EPSILON / 2.0F ? -FLT_EPSILON / 2.0F : logf(value);
    return -log_value;
}

__global__ void sparse_multinomial_exact_kernel(const int32_t* __restrict__ indices,
                                                const float* __restrict__ probabilities,
                                                int32_t rows, int32_t vocab_size, int32_t keep,
                                                uint64_t seed, uint64_t base_offset,
                                                int32_t total_threads,
                                                int32_t* __restrict__ out_token_ids) {
    __shared__ float scores[kSamplerBlockSize];
    __shared__ int32_t tokens[kSamplerBlockSize];

    const int32_t row = static_cast<int32_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int32_t row_offset = row * keep;
    const int thread_id = threadIdx.x;
    float best_score = -FLT_MAX;
    int32_t best_token = 0;

    for (int32_t index = thread_id; index < keep; index += blockDim.x) {
        const int32_t token_id = indices[row_offset + index];
        const int64_t linear_index =
            static_cast<int64_t>(row) * vocab_size + static_cast<int64_t>(token_id);
        const int64_t quotient = linear_index / total_threads;
        const uint64_t loop_iteration = static_cast<uint64_t>(quotient / 4);
        const int component = static_cast<int>(quotient % 4);
        const int64_t subsequence = linear_index % total_threads;

        curandStatePhilox4_32_10_t state;
        curand_init(seed, static_cast<unsigned long long>(subsequence),
                    base_offset + kGeneratorOffsetsPerCurandCall * loop_iteration, &state);
        const float4 random = curand_uniform4(&state);
        const float uniform = component == 0   ? random.x
                              : component == 1 ? random.y
                              : component == 2 ? random.z
                                               : random.w;
        const float score =
            probabilities[row_offset + index] / torch_exponential_from_uniform(uniform);
        if (score > best_score) {
            best_score = score;
            best_token = token_id;
        }
    }

    scores[thread_id] = best_score;
    tokens[thread_id] = best_token;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride && scores[thread_id + stride] > scores[thread_id]) {
            scores[thread_id] = scores[thread_id + stride];
            tokens[thread_id] = tokens[thread_id + stride];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        out_token_ids[row] = tokens[0];
    }
}

} // namespace

void bark_gpu_sparse_torch_multinomial_exact(const int32_t* d_indices, const float* d_probs,
                                             int32_t rows, int32_t vocab_size, int32_t keep,
                                             uint64_t seed, uint64_t base_offset,
                                             int32_t total_threads, int32_t* d_token_ids,
                                             cudaStream_t stream) {
    if (rows <= 0 || vocab_size <= 0 || keep <= 0 || d_indices == nullptr || d_probs == nullptr ||
        d_token_ids == nullptr || total_threads <= 0) {
        return;
    }

    sparse_multinomial_exact_kernel<<<rows, kSamplerBlockSize, 0, stream>>>(
        d_indices, d_probs, rows, vocab_size, keep, seed, base_offset, total_threads, d_token_ids);
}

} // namespace trtmc
