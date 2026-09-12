/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Drives the Marian pipeline's own allocation paths - allocate_cross_kv, which
// the constructor runs, and ensure_encoder_mask, which setup_cross_attention
// runs - against CPU CUDA stubs so each allocation can be failed in turn
// without a GPU. MarianPipeline itself needs live TensorRT modules to build.

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
            trtmc::marian::allocate_cross_kv(keys, values, layers, bytes);
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
        trtmc::marian::allocate_cross_kv(keys, values, layers, bytes);
        check(g_outstanding.size() == static_cast<std::size_t>(2 * layers),
              "every layer should hold a key and a value buffer");
    }
    check(g_outstanding.empty(), "leaving scope should release every buffer");

    // The encoder mask is allocated lazily by setup_cross_attention. A failure
    // must throw and hold nothing; a success must allocate exactly once, and a
    // repeat call must reuse it rather than allocate again.
    {
        trtmc::marian::DeviceBuffer mask;
        g_fail_on_allocation = 1;
        g_allocation_count = 0;
        bool mask_threw = false;
        try {
            trtmc::marian::ensure_encoder_mask(mask, bytes);
        } catch (const std::runtime_error& error) {
            mask_threw = true;
            check(std::string(error.what()) ==
                      "MarianPipeline: unable to allocate encoder mask buffer",
                  "the encoder mask error should name the encoder mask");
        }
        check(mask_threw, "a failed encoder mask allocation should throw");
        check(mask.get() == nullptr && g_outstanding.empty(),
              "a failed encoder mask allocation must hold nothing");

        g_fail_on_allocation = 0;
        g_allocation_count = 0;
        trtmc::marian::ensure_encoder_mask(mask, bytes);
        check(mask.get() != nullptr && g_outstanding.size() == 1,
              "the encoder mask should allocate once");
        trtmc::marian::ensure_encoder_mask(mask, bytes);
        check(g_allocation_count == 1, "a second call must reuse the mask, not reallocate");
    }
    check(g_outstanding.empty(), "leaving scope should release the encoder mask");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("marian cross-attention allocation-failure checks passed\n");
    return 0;
}
