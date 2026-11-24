/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/gemma/runtime/kv_cache.h"
#include "trtmc/internal/config.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// CPU-backed CUDA calls exercise the real cache and DeviceTensor implementations
// without a GPU. An overlapping copy is undefined for cudaMemcpyAsync, so the
// shim records it and refuses to copy.
namespace {
constexpr int32_t kRows = 5;
constexpr int32_t kWidth = 3;
constexpr std::size_t kRowBytes = kWidth * sizeof(float);

struct Allocation {
    void* pointer;
    std::size_t size;
};

std::vector<Allocation> allocations;
int copies = 0;
bool overlapping_copy = false;

void check(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

cudaError_t copy(void* destination, const void* source, std::size_t count) {
    ++copies;
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

// The single-row present buffers are the only one-row allocations; the
// constructor allocates the key buffer before the value buffer.
void write_present(float key, float value) {
    std::vector<void*> rows;
    for (const auto& allocation : allocations) {
        if (allocation.size == kRowBytes)
            rows.push_back(allocation.pointer);
    }
    check(rows.size() == 2, "Expected one present key and one present value row");
    const std::vector<float> key_row(kWidth, key);
    const std::vector<float> value_row(kWidth, value);
    std::memcpy(rows[0], key_row.data(), kRowBytes);
    std::memcpy(rows[1], value_row.data(), kRowBytes);
}

std::vector<float> rows_of(std::initializer_list<float> values) {
    std::vector<float> result;
    for (const float value : values)
        result.insert(result.end(), kWidth, value);
    return result;
}

void test_advance_stops_at_a_full_cache() {
    allocations.clear();
    trtmc::GemmaKvCache cache(1, kRows, kWidth, nullptr);
    check(cache.ok(), "Valid cache allocation was rejected");
    const auto prompt_keys = rows_of({1, 2, 3, 4});
    const auto prompt_values = rows_of({-1, -2, -3, -4});
    cache.write_prefill_kv({prompt_keys.data()}, {prompt_values.data()}, 4);
    check(cache.position() == 4, "Prefill did not fill four rows");

    write_present(5, -5);
    cache.advance();
    check(cache.position() == kRows, "Advance did not append the decoded row");
    check(read(cache.cache_k(0)) == rows_of({1, 2, 3, 4, 5}),
          "Advance wrote the keys to the wrong row");
    check(read(cache.cache_v(0)) == rows_of({-1, -2, -3, -4, -5}),
          "Advance wrote the values to the wrong row");

    // A full cache never drops its oldest rows: the next advance is rejected
    // before any copy, and the cached prompt is unchanged.
    write_present(6, -6);
    const int copies_before = copies;
    bool rejected = false;
    try {
        cache.advance();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "Advancing a full cache must fail explicitly");
    check(copies == copies_before, "A rejected advance issued a CUDA copy");
    check(cache.position() == kRows, "A rejected advance changed the logical position");
    check(read(cache.cache_k(0)) == rows_of({1, 2, 3, 4, 5}),
          "A rejected advance changed the cached keys");
    check(read(cache.cache_v(0)) == rows_of({-1, -2, -3, -4, -5}),
          "A rejected advance changed the cached values");
    check(!overlapping_copy, "The cache performed an overlapping CUDA copy");

    cache.reset();
    cache.advance();
    check(cache.position() == 1, "The cache did not recover after reset");
}

void test_prompt_capacity() {
    trtmc::GemmaKvCache cache(1, kRows, kWidth, nullptr);
    cache.set_prompt_length(kRows);
    try {
        cache.set_prompt_length(kRows + 1);
    } catch (const trtmc::internal::ConfigError& error) {
        // ConfigError is the request-rejection type: the C API reports it as
        // TRTMC_INVALID_CONFIG and a serving worker answers a request error.
        const std::string message = error.what();
        check(message.find("6 tokens") != std::string::npos &&
                  message.find("holds 5") != std::string::npos &&
                  message.find("--max-sequence-length") != std::string::npos,
              "Prompt overflow error does not name both lengths and the build option: " + message);
        return;
    }
    throw std::runtime_error("A prompt longer than the KV cache must be rejected as invalid");
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
        test_advance_stops_at_a_full_cache();
        test_prompt_capacity();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
