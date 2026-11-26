/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/nemotron_voicechat/runtime/thinker_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// CPU-backed CUDA calls exercise the real thinker cache and DeviceTensor
// implementations without a GPU. An overlapping copy is undefined for
// cudaMemcpyAsync, so the shim records it and refuses to copy.
namespace {
constexpr int32_t kRows = 3;
constexpr int32_t kWidth = 2;
constexpr std::size_t kRowBytes = kWidth * sizeof(float);
constexpr std::size_t kCacheBytes = kRows * kRowBytes;

struct Allocation {
    void* pointer;
    std::size_t size;
};

std::vector<Allocation> allocations;
bool overlapping_copy = false;

void check(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

cudaError_t copy(void* destination, const void* source, std::size_t count) {
    if (!count)
        return cudaSuccess;
    if (!destination || !source)
        return cudaErrorInvalidValue;
    const auto dst = reinterpret_cast<std::uintptr_t>(destination);
    const auto src = reinterpret_cast<std::uintptr_t>(source);
    if (dst < src + count && src < dst + count) {
        overlapping_copy = true;
        return cudaErrorInvalidValue;
    }
    std::memcpy(destination, source, count);
    return cudaSuccess;
}

// The constructor allocates cache_k, cache_v, present_k and present_v; with one
// layer they are told apart by size, keys before values.
std::vector<float*> buffers(std::size_t size) {
    std::vector<float*> result;
    for (const auto& allocation : allocations) {
        if (allocation.size == size)
            result.push_back(static_cast<float*>(allocation.pointer));
    }
    check(result.size() == 2, "Expected one key and one value buffer of each size");
    return result;
}

std::vector<float> rows_of(std::initializer_list<float> values) {
    std::vector<float> result;
    for (const float value : values)
        result.insert(result.end(), kWidth, value);
    return result;
}

std::vector<float> contents(const float* buffer) {
    return std::vector<float>(buffer, buffer + kRows * kWidth);
}

std::vector<float> mask_of(trtmc::VoiceChatThinkerKvCache& cache) {
    trtmc::TensorMap inputs;
    cache.prepare_step(inputs);
    const auto it = inputs.find("attention_mask");
    check(it != inputs.end(), "prepare_step did not write the attention mask");
    const auto* values = static_cast<const float*>(it->second.data);
    return std::vector<float>(values, values + kRows + 1);
}

void test_full_cache_keeps_latest_rows() {
    allocations.clear();
    trtmc::VoiceChatThinkerKvCache cache(1, kRows, kWidth, nullptr);
    check(cache.ok(), "Valid cache allocation was rejected");
    const auto present = buffers(kRowBytes);
    const auto cached = buffers(kCacheBytes);

    constexpr float kMasked = -1.0e4F;
    check(mask_of(cache) == std::vector<float>{kMasked, kMasked, kMasked, 0},
          "An empty cache must show only the current token");
    for (int32_t step = 1; step <= 5; ++step) {
        const std::vector<float> key(kWidth, static_cast<float>(step));
        const std::vector<float> value(kWidth, static_cast<float>(-step));
        std::memcpy(present[0], key.data(), kRowBytes);
        std::memcpy(present[1], value.data(), kRowBytes);
        cache.advance();
    }

    // Steps 4 and 5 replaced the two oldest rows in place; every row is visible.
    check(contents(cached[0]) == rows_of({4, 5, 3}), "The ring kept the wrong keys");
    check(contents(cached[1]) == rows_of({-4, -5, -3}), "The ring kept the wrong values");
    check(mask_of(cache) == std::vector<float>{0, 0, 0, 0}, "A full ring must show every row");
    check(!overlapping_copy, "The cache performed an overlapping CUDA copy");

    cache.reset();
    check(mask_of(cache) == std::vector<float>{kMasked, kMasked, kMasked, 0},
          "Reset must hide every cached row");
    const std::vector<float> key(kWidth, 9.0F);
    std::memcpy(present[0], key.data(), kRowBytes);
    cache.advance();
    check(contents(cached[0]) == std::vector<float>{9, 9, 5, 5, 3, 3},
          "After reset the ring must write from its first row again");
    check(mask_of(cache) == std::vector<float>{0, kMasked, kMasked, 0},
          "After reset only the new row is visible");
}
} // namespace

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** pointer, std::size_t size) {
    *pointer = std::malloc(size);
    if (!*pointer)
        return cudaErrorMemoryAllocation;
    std::memset(*pointer, 0xff, size);
    allocations.push_back({*pointer, size});
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaFree(void* pointer) {
    std::free(pointer);
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaMemcpy(void* destination, const void* source,
                                            std::size_t count, cudaMemcpyKind) {
    return copy(destination, source, count);
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void* destination, const void* source,
                                                 std::size_t count, cudaMemcpyKind, cudaStream_t) {
    return copy(destination, source, count);
}
extern "C" cudaError_t CUDARTAPI cudaMemsetAsync(void* destination, int value, std::size_t count,
                                                 cudaStream_t) {
    if (!destination)
        return cudaErrorInvalidValue;
    std::memset(destination, value, count);
    return cudaSuccess;
}

int main() {
    try {
        test_full_cache_keeps_latest_rows();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
