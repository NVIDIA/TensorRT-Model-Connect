/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llava/runtime/kv_cache.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

// CPU-backed CUDA calls exercise the real cache and DeviceTensor implementations,
// including deterministic allocation failures, without requiring a GPU.
namespace {
int allocation_count = 0;
int fail_allocation = -1;
bool overlapping_copy = false;

void check(bool condition, const char* message) {
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

void test_zero_storage() {
    for (auto dtype : {trtmc::DType::kFloat16, trtmc::DType::kFloat32}) {
        trtmc::LlavaKvCache cache(2, 4, 2, nullptr, dtype);
        check(cache.ok(), "Valid cache allocation was rejected");
        for (int layer = 0; layer < 2; ++layer) {
            for (auto* tensor : {&cache.cache_k(layer), &cache.cache_v(layer)}) {
                std::vector<unsigned char> bytes(tensor->nbytes());
                check(tensor->copy_to_host(bytes.data()), "Cache read failed");
                check(
                    std::all_of(bytes.begin(), bytes.end(), [](auto value) { return value == 0; }),
                    "Unwritten cache storage must start at zero");
            }
        }
    }
}

void test_exhaustion() {
    trtmc::LlavaKvCache cache(1, 4, 2, nullptr);
    const std::vector<float> key{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> value{-1, -2, -3, -4, -5, -6, -7, -8};
    cache.write_prefill_kv({key.data()}, {value.data()}, 4);
    bool rejected = false;
    try {
        cache.advance();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "Cache exhaustion must fail explicitly");
    check(!overlapping_copy, "Cache performed an overlapping CUDA copy");
    check(cache.position() == 4, "Rejected append changed the logical position");
    std::vector<float> observed(8);
    check(cache.cache_k(0).copy_to_host(observed.data()) && observed == key,
          "Rejected append changed cached keys");
    check(cache.cache_v(0).copy_to_host(observed.data()) && observed == value,
          "Rejected append changed cached values");
    cache.reset();
    cache.append_prefill_kv({key.data()}, {value.data()}, 2);
    check(cache.position() == 2, "Cache did not recover after reset");
}

void test_allocation_failures() {
    for (int failure = 0; failure < 8; ++failure) {
        allocation_count = 0;
        fail_allocation = failure;
        trtmc::LlavaKvCache cache(2, 4, 2, nullptr);
        check(!cache.ok(), "A K/V or present-buffer allocation failure was missed");
    }
    fail_allocation = -1;
    trtmc::LlavaKvCache valid(2, 4, 2, nullptr);
    check(valid.ok(), "Healthy cache allocation failed");
}
} // namespace

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** pointer, std::size_t size) {
    if (allocation_count++ == fail_allocation) {
        *pointer = nullptr;
        return cudaErrorMemoryAllocation;
    }
    *pointer = std::malloc(size);
    if (!*pointer)
        return cudaErrorMemoryAllocation;
    std::memset(*pointer, 0xff, size);
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

int main(int argc, char** argv) {
    try {
        check(argc == 2, "Expected zero, exhaustion, or allocation test mode");
        const std::string mode(argv[1]);
        if (mode == "zero")
            test_zero_storage();
        else if (mode == "exhaustion")
            test_exhaustion();
        else if (mode == "allocation")
            test_allocation_failures();
        else
            throw std::runtime_error("Unknown cache test mode");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
