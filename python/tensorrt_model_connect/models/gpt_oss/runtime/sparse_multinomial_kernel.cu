/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sparse_multinomial_kernel.h"

#include <curand_kernel.h>

#include <algorithm>
#include <cfloat>
#include <limits>

namespace trtmc {

namespace {

constexpr int kDistributionBlockSize = 256;
constexpr int kSamplerBlockSize = 128;
constexpr uint64_t kGeneratorOffsetsPerCurandCall = 4;

__device__ float torch_exponential_from_uniform(float val) {
    const float log_val = val >= 1.0F - FLT_EPSILON / 2.0F ? -FLT_EPSILON / 2.0F : logf(val);
    return -log_val;
}

__global__ void sparse_multinomial_exact_kernel(const int32_t* __restrict__ indices,
                                                const float* __restrict__ probs, int32_t keep,
                                                uint64_t seed, uint64_t base_offset,
                                                int32_t total_threads,
                                                int32_t* __restrict__ out_token_id) {
    __shared__ float s_scores[kSamplerBlockSize];
    __shared__ int32_t s_tokens[kSamplerBlockSize];

    const int tid = threadIdx.x;
    float best_score = -FLT_MAX;
    int32_t best_token = 0;

    for (int32_t i = tid; i < keep; i += blockDim.x) {
        const int32_t token_id = indices[i];
        const int64_t linear_index = static_cast<int64_t>(token_id);
        const int64_t q = linear_index / total_threads;
        const uint64_t loop_iteration = static_cast<uint64_t>(q / 4);
        const int component = static_cast<int>(q % 4);
        const int64_t subsequence = linear_index % total_threads;

        curandStatePhilox4_32_10_t state;
        curand_init(seed, static_cast<unsigned long long>(subsequence),
                    base_offset + kGeneratorOffsetsPerCurandCall * loop_iteration, &state);
        const float4 rand = curand_uniform4(&state);
        const float uniform = component == 0 ? rand.x : component == 1 ? rand.y : component == 2 ? rand.z : rand.w;
        const float score = probs[i] / torch_exponential_from_uniform(uniform);
        if (score > best_score) {
            best_score = score;
            best_token = token_id;
        }
    }

    s_scores[tid] = best_score;
    s_tokens[tid] = best_token;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_scores[tid + stride] > s_scores[tid]) {
            s_scores[tid] = s_scores[tid + stride];
            s_tokens[tid] = s_tokens[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_token_id = s_tokens[0];
    }
}

} // namespace

GptOssTorchMultinomialExecutionPolicy gpt_oss_compute_torch_multinomial_execution_policy(int32_t numel) {
    if (numel <= 0) {
        return {};
    }

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, device);

    const uint32_t blocks_per_sm =
        static_cast<uint32_t>(props.maxThreadsPerMultiProcessor / kDistributionBlockSize);
    const uint32_t grid = std::min(
        static_cast<uint32_t>(props.multiProcessorCount) * blocks_per_sm,
        static_cast<uint32_t>((static_cast<uint64_t>(numel) + kDistributionBlockSize - 1)
                              / kDistributionBlockSize));
    const uint64_t total_threads = static_cast<uint64_t>(grid) * kDistributionBlockSize;
    const uint64_t counter_offset =
        ((static_cast<uint64_t>(numel) - 1)
         / (total_threads * kGeneratorOffsetsPerCurandCall) + 1)
        * kGeneratorOffsetsPerCurandCall;

    GptOssTorchMultinomialExecutionPolicy policy;
    policy.total_threads = static_cast<int32_t>(total_threads);
    policy.counter_offset = counter_offset;
    return policy;
}

void gpt_oss_gpu_sparse_torch_multinomial_exact(const int32_t* d_indices, const float* d_probs,
                                        int32_t keep, uint64_t seed, uint64_t base_offset,
                                        int32_t total_threads, int32_t* d_token_id,
                                        cudaStream_t stream) {
    if (keep <= 0 || d_indices == nullptr || d_probs == nullptr || d_token_id == nullptr
        || total_threads <= 0) {
        return;
    }

    sparse_multinomial_exact_kernel<<<1, kSamplerBlockSize, 0, stream>>>(
        d_indices, d_probs, keep, seed, base_offset, total_threads, d_token_id);
}

} // namespace trtmc
