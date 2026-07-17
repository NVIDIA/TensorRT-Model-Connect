/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/torch_cuda_normal.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <utility>
#include <vector>

namespace {

uint32_t bits(float value) {
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

uint64_t fnv1a(const std::vector<float>& values) {
    uint64_t hash = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
    for (std::size_t index = 0; index < values.size() * sizeof(float); ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

int main() {
    const auto values = trtmc::wan2_2_ti2v::torch_cuda_normal(1024, 42);
    const std::vector<std::pair<std::size_t, uint32_t>> expected = {
        {0, 0x3e46acddU},    {1, 0x400a53f2U},    {2, 0xbe302defU},    {3, 0x3f595c01U},
        {4, 0xbff652b5U},    {5, 0x3f272a0fU},    {6, 0xbf2641c1U},    {7, 0xbf51494dU},
        {8, 0x3f0728aeU},    {9, 0xbfa33eaaU},    {10, 0xbfd4c08eU},   {11, 0xbe9b4bf0U},
        {12, 0xbdbd9546U},   {13, 0x3e4c04d3U},   {14, 0xbf8f6a58U},   {15, 0x3fedc7c2U},
        {1020, 0x3f878de6U}, {1021, 0x4036cf23U}, {1022, 0x3f2cc155U}, {1023, 0x400398e2U},
    };
    int failures = 0;
    for (const auto& [index, expected_bits] : expected) {
        if (bits(values[index]) != expected_bits) {
            std::cerr << "FAIL: seed-42 torch CUDA normal mismatch at " << index << " actual=0x"
                      << std::hex << bits(values[index]) << " expected=0x" << expected_bits
                      << std::dec << '\n';
            ++failures;
        }
    }

    const auto time_features = trtmc::wan2_2_ti2v::torch_cuda_timestep_features(999);
    constexpr uint64_t expected_time_hash = 0xb652c69cfbefd486ULL;
    if (fnv1a(time_features) != expected_time_hash) {
        std::cerr << "FAIL: t=999 CUDA timestep-feature FNV-1a mismatch actual=0x" << std::hex
                  << fnv1a(time_features) << " expected=0x" << expected_time_hash << std::dec
                  << '\n';
        ++failures;
    }

    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
        properties.multiProcessorCount == 152 && properties.maxThreadsPerMultiProcessor == 2048) {
        constexpr std::size_t official_count = 48ULL * 31ULL * 44ULL * 80ULL;
        const auto full = trtmc::wan2_2_ti2v::torch_cuda_normal(official_count, 42);
        constexpr uint64_t expected_hash = 0x1f3097843fedc581ULL;
        if (fnv1a(full) != expected_hash) {
            std::cerr << "FAIL: full GB300 seed-42 Wan2.2 latent FNV-1a mismatch actual=0x"
                      << std::hex << fnv1a(full) << " expected=0x" << expected_hash << std::dec
                      << '\n';
            ++failures;
        }
    } else {
        std::cerr << "SKIP: full latent golden is GB300-specific; portable 1024-value probe ran\n";
    }
    return failures == 0 ? 0 : 1;
}
