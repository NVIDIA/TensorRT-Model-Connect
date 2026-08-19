/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "torch_cuda_normal.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::wan2_2_ti2v {
namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kUnroll = 4;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Wan2.2 CUDA RNG ") + operation +
                                 " failed: " + cudaGetErrorString(status));
}

__global__ __launch_bounds__(kBlockSize, 4) void torch_normal_kernel(int64_t count, uint64_t seed,
                                                                     float* output) {
    const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<uint64_t>(thread), 0, &state);

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t group = stride * kUnroll;
    const int64_t rounded = ((count - 1) / group + 1) * group;
    for (int64_t linear = thread; linear < rounded; linear += group) {
        const float4 values = curand_normal4(&state);
        const float* source = &values.x;
#pragma unroll
        for (uint32_t lane = 0; lane < kUnroll; ++lane) {
            const int64_t index = linear + stride * lane;
            if (index < count)
                output[index] = source[lane];
        }
    }
}

__global__ void torch_timestep_features_kernel(int64_t timestep, float* output) {
    const int32_t index = static_cast<int32_t>(threadIdx.x);
    const double exponent = -static_cast<double>(index) / 128.0;
    const double frequency = pow(10000.0, exponent);
    const double phase = static_cast<double>(timestep) * frequency;
    output[index] = static_cast<float>(cos(phase));
    output[128 + index] = static_cast<float>(sin(phase));
}

} // namespace

std::vector<float> torch_cuda_normal(std::size_t count, uint64_t seed) {
    if (count == 0)
        return {};
    if (count > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("Wan2.2 CUDA RNG tensor is too large");

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    const auto requested_blocks = (count + kBlockSize - 1U) / kBlockSize;
    const auto blocks_per_sm =
        static_cast<std::size_t>(properties.maxThreadsPerMultiProcessor) / kBlockSize;
    const auto resident_blocks =
        static_cast<std::size_t>(properties.multiProcessorCount) * blocks_per_sm;
    const auto grid = static_cast<uint32_t>(std::min(requested_blocks, resident_blocks));
    if (grid == 0)
        throw std::runtime_error("Wan2.2 CUDA RNG computed an empty launch grid");

    float* device_output = nullptr;
    check_cuda(cudaMalloc(&device_output, count * sizeof(float)), "cudaMalloc");
    try {
        torch_normal_kernel<<<grid, kBlockSize>>>(static_cast<int64_t>(count), seed, device_output);
        check_cuda(cudaGetLastError(), "kernel launch");
        std::vector<float> output(count);
        check_cuda(
            cudaMemcpy(output.data(), device_output, count * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy");
        check_cuda(cudaFree(device_output), "cudaFree");
        return output;
    } catch (...) {
        cudaFree(device_output);
        throw;
    }
}

std::vector<float> torch_cuda_timestep_features(int64_t timestep) {
    constexpr std::size_t count = 256;
    float* device_output = nullptr;
    check_cuda(cudaMalloc(&device_output, count * sizeof(float)), "timestep cudaMalloc");
    try {
        torch_timestep_features_kernel<<<1, 128>>>(timestep, device_output);
        check_cuda(cudaGetLastError(), "timestep kernel launch");
        std::vector<float> output(count);
        check_cuda(
            cudaMemcpy(output.data(), device_output, count * sizeof(float), cudaMemcpyDeviceToHost),
            "timestep cudaMemcpy");
        check_cuda(cudaFree(device_output), "timestep cudaFree");
        return output;
    } catch (...) {
        cudaFree(device_output);
        throw;
    }
}

} // namespace trtmc::wan2_2_ti2v
