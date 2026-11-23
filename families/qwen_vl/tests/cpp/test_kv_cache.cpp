/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen_vl/runtime/kv_cache.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// CPU-backed CUDA calls exercise the real cache and DeviceTensor implementations
// without a GPU. An overlapping copy is undefined for cudaMemcpyAsync, so the
// shim records it and refuses to copy.
namespace {
std::vector<void*> allocations;
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

std::vector<float> read(const trtmc::DeviceTensor& tensor) {
    std::vector<float> values(tensor.nbytes() / sizeof(float));
    check(tensor.copy_to_host(values.data()), "Cache read failed");
    return values;
}

// The constructor allocates cache_k, cache_v, present_k and present_v for each
// layer in that order, so with one layer the present rows are allocations 2 and 3.
void write_present(const std::vector<float>& key, const std::vector<float>& value) {
    check(allocations.size() >= 4, "Present buffers were not allocated");
    std::memcpy(allocations[2], key.data(), key.size() * sizeof(float));
    std::memcpy(allocations[3], value.data(), value.size() * sizeof(float));
}

void test_full_cache_shift() {
    allocations.clear();
    trtmc::QwenVlKvCache cache(1, 4, 2, nullptr);
    check(cache.ok(), "Valid cache allocation was rejected");
    const std::vector<float> key{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> value{-1, -2, -3, -4, -5, -6, -7, -8};
    cache.write_prefill_kv({key.data()}, {value.data()}, 4);

    write_present({9, 10}, {-9, -10});
    cache.advance();
    check(!overlapping_copy, "Cache shift performed an overlapping CUDA copy");
    check(cache.position() == 4, "A full cache must keep every slot visible");
    check(read(cache.cache_k(0)) == std::vector<float>{3, 4, 5, 6, 7, 8, 9, 10},
          "Full-cache shift corrupted the keys");
    check(read(cache.cache_v(0)) == std::vector<float>{-3, -4, -5, -6, -7, -8, -9, -10},
          "Full-cache shift corrupted the values");

    write_present({11, 12}, {-11, -12});
    cache.advance();
    check(!overlapping_copy, "Repeated cache shift performed an overlapping CUDA copy");
    check(read(cache.cache_k(0)) == std::vector<float>{5, 6, 7, 8, 9, 10, 11, 12},
          "Repeated full-cache shift corrupted the keys");
    check(read(cache.cache_v(0)) == std::vector<float>{-5, -6, -7, -8, -9, -10, -11, -12},
          "Repeated full-cache shift corrupted the values");
}

void test_prompt_capacity() {
    trtmc::QwenVlKvCache cache(1, 4, 2, nullptr);
    cache.set_prompt_length(4);
    try {
        cache.set_prompt_length(5);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        check(message.find("5 tokens") != std::string::npos &&
                  message.find("holds 4") != std::string::npos &&
                  message.find("--max-sequence-length") != std::string::npos,
              "Prompt overflow error does not name both lengths and the build option: " + message);
        return;
    }
    throw std::runtime_error("A prompt longer than the KV cache must be rejected");
}
} // namespace

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** pointer, std::size_t size) {
    *pointer = std::malloc(size);
    if (!*pointer)
        return cudaErrorMemoryAllocation;
    std::memset(*pointer, 0xff, size);
    allocations.push_back(*pointer);
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
        test_full_cache_shift();
        test_prompt_capacity();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
