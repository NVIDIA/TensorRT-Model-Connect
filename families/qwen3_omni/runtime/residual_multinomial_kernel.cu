/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_omni/runtime/residual_multinomial_kernel.h"

#include <cfloat>
#include <curand_kernel.h>

namespace trtmc::qwen3_omni {
namespace {

constexpr int kBlockSize = 128;
constexpr std::uint64_t kOffsetsPerDraw = 4;

__device__ float torch_exponential_from_uniform(float value) {
    const float log_value = value >= 1.0F - FLT_EPSILON / 2.0F ? -FLT_EPSILON / 2.0F : logf(value);
    return -log_value;
}

__global__ void residual_exponential_race_kernel(const float* __restrict__ probabilities,
                                                 std::size_t count, std::uint64_t seed,
                                                 std::uint64_t draw,
                                                 std::int32_t* __restrict__ output_token) {
    __shared__ float scores[kBlockSize];
    __shared__ std::int32_t tokens[kBlockSize];

    const int thread = threadIdx.x;
    float best_score = -FLT_MAX;
    std::int32_t best_token = 0;
    for (std::size_t token = static_cast<std::size_t>(thread); token < count; token += blockDim.x) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, static_cast<unsigned long long>(token), kOffsetsPerDraw * draw, &state);
        const float uniform = curand_uniform4(&state).x;
        const float score = probabilities[token] / torch_exponential_from_uniform(uniform);
        if (score > best_score) {
            best_score = score;
            best_token = static_cast<std::int32_t>(token);
        }
    }

    scores[thread] = best_score;
    tokens[thread] = best_token;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride && scores[thread + stride] > scores[thread]) {
            scores[thread] = scores[thread + stride];
            tokens[thread] = tokens[thread + stride];
        }
        __syncthreads();
    }
    if (thread == 0)
        *output_token = tokens[0];
}

} // namespace

void launch_residual_exponential_race(const float* device_probabilities, std::size_t count,
                                      std::uint64_t seed, std::uint64_t draw,
                                      std::int32_t* device_token, cudaStream_t stream) {
    residual_exponential_race_kernel<<<1, kBlockSize, 0, stream>>>(device_probabilities, count,
                                                                   seed, draw, device_token);
}

} // namespace trtmc::qwen3_omni
