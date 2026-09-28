/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/turbo_noise.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::minimax_h3 {
namespace {

constexpr std::size_t kBlockThreads = 256;
constexpr std::size_t kLanes = 4;
using Words = std::array<std::uint32_t, kLanes>;

// Philox4x32-10 (Random123): a seed is the two-word key, a CUDA thread's
// subsequence occupies the upper counter words, and each normal4 call advances
// the lower 64-bit counter by one. Implemented with standard integer arithmetic.
Words philox(std::uint64_t seed, std::uint64_t subsequence, std::uint64_t block) {
    Words counter{static_cast<std::uint32_t>(block), static_cast<std::uint32_t>(block >> 32),
                  static_cast<std::uint32_t>(subsequence),
                  static_cast<std::uint32_t>(subsequence >> 32)};
    std::uint32_t key_low = static_cast<std::uint32_t>(seed);
    std::uint32_t key_high = static_cast<std::uint32_t>(seed >> 32);
    for (unsigned round = 0; round < 10; ++round) {
        const std::uint64_t first = std::uint64_t{0xd2511f53U} * counter[0];
        const std::uint64_t second = std::uint64_t{0xcd9e8d57U} * counter[2];
        counter = {static_cast<std::uint32_t>(second >> 32) ^ counter[1] ^ key_low,
                   static_cast<std::uint32_t>(second),
                   static_cast<std::uint32_t>(first >> 32) ^ counter[3] ^ key_high,
                   static_cast<std::uint32_t>(first)};
        key_low += 0x9e3779b9U;
        key_high += 0xbb67ae85U;
    }
    return counter;
}

float round_bfloat16(float value) {
    std::uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    if ((word & 0x7fffffffU) > 0x7f800000U) {
        word = (word & 0xffff0000U) | 0x00400000U;
    } else {
        word = (word + 0x7fffU + ((word >> 16) & 1U)) & 0xffff0000U;
    }
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::array<float, kLanes> normal4(const Words& words) {
    // Same midpoint-uniform Box-Muller transform and sine/cosine lane order as
    // NVIDIA's published curand_normal.h, independently expressed on the host.
    // https://docs.nvidia.com/cuda/curand/curand/curand__normal_8h_source.html
    constexpr float uniform_scale = 0x1p-32F;
    constexpr float angle_scale = 1.4629180792671596e-9F;
    std::array<float, kLanes> values;
    for (std::size_t pair = 0; pair < kLanes; pair += 2) {
        const float uniform = std::fma(static_cast<float>(words[pair]), uniform_scale,
                                       uniform_scale * 0.5F);
        const float angle = std::fma(static_cast<float>(words[pair + 1]), angle_scale,
                                     angle_scale * 0.5F);
        const float radius = std::sqrt(-2.0F * std::log(uniform));
        values[pair] = round_bfloat16(std::sin(angle) * radius);
        values[pair + 1] = round_bfloat16(std::cos(angle) * radius);
    }
    return values;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("MiniMax-H3 Turbo noise ") + operation + ": " +
                                 cudaGetErrorString(status));
}

} // namespace

std::vector<float> make_minimax_h3_turbo_noise(std::size_t count, std::uint64_t seed) {
    if (count == 0)
        return {};
    // Torch's 32-bit TensorIterator path is the one reproduced below. Its large
    // tensor subdivision has different generator-offset semantics; fail closed.
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) / 2)
        throw std::invalid_argument("MiniMax-H3 Turbo noise exceeds the BF16 indexing range");

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (properties.multiProcessorCount <= 0 || properties.maxThreadsPerMultiProcessor < 256)
        throw std::runtime_error("MiniMax-H3 Turbo noise received invalid CUDA launch limits");
    const std::size_t max_blocks = static_cast<std::size_t>(properties.multiProcessorCount) *
        static_cast<std::size_t>(properties.maxThreadsPerMultiProcessor / 256);
    const std::size_t blocks = std::min((count + kBlockThreads - 1) / kBlockThreads, max_blocks);
    const std::size_t threads = blocks * kBlockThreads;
    const std::size_t group_stride = threads * kLanes;
    std::vector<float> output(count);

    // Mirrors PyTorch 2.11 ATen/native/cuda/DistributionTemplates.h's contiguous
    // distribution_elementwise_grid_stride_kernel with a float4 distribution.
    for (std::size_t start = 0, block = 0; start < count; start += group_stride, ++block) {
        const std::size_t active_threads = std::min(threads, count - start);
        for (std::size_t thread = 0; thread < active_threads; ++thread) {
            const auto values = normal4(philox(seed, thread, block));
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const std::size_t index = start + thread + lane * threads;
                if (index < count)
                    output[index] = values[lane];
            }
        }
    }
    return output;
}

} // namespace trtmc::minimax_h3
