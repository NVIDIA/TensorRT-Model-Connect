/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bark/runtime/sparse_multinomial_kernel.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace trtmc {

namespace {

constexpr int kDistributionBlockSize = 256;

} // namespace

BarkTorchMultinomialExecutionPolicy bark_compute_torch_multinomial_execution_policy(int32_t numel) {
    if (numel <= 0) {
        return {};
    }

    int device = 0;
    const cudaError_t device_status = cudaGetDevice(&device);
    if (device_status != cudaSuccess) {
        throw std::runtime_error("cudaGetDevice failed for the bark multinomial launch: " +
                                 std::string(cudaGetErrorString(device_status)));
    }

    cudaDeviceProp properties{};
    const cudaError_t properties_status = cudaGetDeviceProperties(&properties, device);
    if (properties_status != cudaSuccess) {
        throw std::runtime_error(
            "cudaGetDeviceProperties failed for the bark multinomial launch: " +
            std::string(cudaGetErrorString(properties_status)));
    }

    const uint32_t blocks_per_sm =
        static_cast<uint32_t>(properties.maxThreadsPerMultiProcessor / kDistributionBlockSize);
    const uint32_t grid =
        std::min(static_cast<uint32_t>(properties.multiProcessorCount) * blocks_per_sm,
                 static_cast<uint32_t>((static_cast<uint64_t>(numel) + kDistributionBlockSize - 1) /
                                       kDistributionBlockSize));
    const uint64_t total_threads = static_cast<uint64_t>(grid) * kDistributionBlockSize;
    // A query can succeed and still report no usable occupancy.
    if (total_threads == 0) {
        throw std::runtime_error("bark multinomial launch policy computed no threads");
    }

    const uint64_t counter_offset =
        ((static_cast<uint64_t>(numel) - 1) / (total_threads * kGeneratorOffsetsPerCurandCall) +
         1) *
        kGeneratorOffsetsPerCurandCall;

    BarkTorchMultinomialExecutionPolicy policy;
    policy.total_threads = static_cast<int32_t>(total_threads);
    policy.counter_offset = counter_offset;
    return policy;
}

} // namespace trtmc
