/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Exercises the ownership the Marian pipeline uses for its cross-attention K/V
// buffers, against CPU CUDA stubs so each allocation can be failed in turn
// without a GPU. MarianPipeline itself needs live TensorRT modules to build, so
// this drives the same allocation loop over the same owning type.

#include "families/marian/runtime/device_buffer.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_fail_on_allocation = 0;
int g_allocation_count = 0;
std::set<void*> g_outstanding;
std::uintptr_t g_next_address = 0x1000;
int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// The allocation loop from MarianPipeline's constructor, over the same type.
void allocate_cross_kv(std::vector<trtmc::marian::DeviceBuffer>& keys,
                       std::vector<trtmc::marian::DeviceBuffer>& values, int32_t layers,
                       std::size_t bytes) {
    keys.resize(static_cast<std::size_t>(layers));
    values.resize(static_cast<std::size_t>(layers));
    for (int32_t i = 0; i < layers; ++i) {
        const std::size_t layer = static_cast<std::size_t>(i);
        if (keys[layer].allocate(bytes) != cudaSuccess)
            throw std::runtime_error(
                "MarianPipeline: unable to allocate cross-attention key buffer");
        if (values[layer].allocate(bytes) != cudaSuccess)
            throw std::runtime_error(
                "MarianPipeline: unable to allocate cross-attention value buffer");
    }
}

} // namespace

extern "C" {

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    (void)size;
    ++g_allocation_count;
    if (g_fail_on_allocation != 0 && g_allocation_count == g_fail_on_allocation) {
        *devPtr = nullptr;
        return cudaErrorMemoryAllocation;
    }
    void* address = reinterpret_cast<void*>(g_next_address);
    g_next_address += 0x1000;
    g_outstanding.insert(address);
    *devPtr = address;
    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    if (devPtr != nullptr) {
        g_outstanding.erase(devPtr);
    }
    return cudaSuccess;
}

} // extern "C"

int main() {
    const int32_t layers = 4;
    const std::size_t bytes = 1024;

    // Four layers means eight allocations. Fail each in turn; whatever the loop
    // acquired before the failure must be released as the exception unwinds.
    for (int failing = 1; failing <= 2 * layers; ++failing) {
        g_fail_on_allocation = failing;
        g_allocation_count = 0;
        bool threw = false;
        try {
            std::vector<trtmc::marian::DeviceBuffer> keys;
            std::vector<trtmc::marian::DeviceBuffer> values;
            allocate_cross_kv(keys, values, layers, bytes);
        } catch (const std::runtime_error& error) {
            threw = true;
            // Allocations alternate key, value, so an odd failure point is a
            // key buffer and an even one is a value buffer.
            const char* expected =
                (failing % 2) == 1
                    ? "MarianPipeline: unable to allocate cross-attention key buffer"
                    : "MarianPipeline: unable to allocate cross-attention value buffer";
            check(std::string(error.what()) == expected,
                  "the error should name the buffer that actually failed");
        }
        check(threw, "a failed cross-attention allocation should throw");
        check(g_outstanding.empty(),
              "a failed allocation must release everything already acquired");
        g_outstanding.clear();
    }

    // With no injected failure the buffers are held, then released on scope exit.
    g_fail_on_allocation = 0;
    g_allocation_count = 0;
    {
        std::vector<trtmc::marian::DeviceBuffer> keys;
        std::vector<trtmc::marian::DeviceBuffer> values;
        allocate_cross_kv(keys, values, layers, bytes);
        check(g_outstanding.size() == static_cast<std::size_t>(2 * layers),
              "every layer should hold a key and a value buffer");
    }
    check(g_outstanding.empty(), "leaving scope should release every buffer");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("marian cross-attention allocation-failure checks passed\n");
    return 0;
}
