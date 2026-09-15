/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Exercises flux::ensure_buf's grow-and-retry contract against CPU CUDA
// stubs, so an allocation failure can be injected without a GPU. The matmul
// scratch buffers are process-lifetime globals reused across calls, unlike
// the per-request buffers the other families own, so what matters here is
// that a failed grow releases the stale buffer and leaves the tracked size
// unchanged, rather than the constructor-unwind behavior those test.

#include "families/flux/runtime/device_buffer.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <set>
#include <stdexcept>

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
    using trtmc::flux::ensure_buf;
    using trtmc::flux::GrowableBuffer;

    // A no-op grow (need <= bytes) must not touch the allocator at all.
    {
        GrowableBuffer gb;
        g_allocation_count = 0;
        ensure_buf(gb, 256);
        check(g_allocation_count == 1, "the first grow from empty should allocate once");
        const int32_t first_count = g_allocation_count;
        ensure_buf(gb, 128);
        check(g_allocation_count == first_count,
              "shrinking the request below the current capacity should not reallocate");
    }
    check(g_outstanding.empty(), "leaving scope should release the buffer");

    // A failed grow must release the stale buffer, leave the pointer null,
    // and leave `bytes` unchanged so the next call retries rather than
    // treating the missing buffer as already sized.
    {
        GrowableBuffer gb;
        ensure_buf(gb, 256);
        check(gb.bytes == 256, "a successful grow should record the new size");

        g_fail_on_allocation = g_allocation_count + 1;
        bool threw = false;
        try {
            ensure_buf(gb, 1024);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "a failed grow should throw");
        check(gb.buf.get() == nullptr, "a failed grow must leave the pointer null, not stale");
        check(gb.bytes == 256, "a failed grow must not update the tracked size");
        check(g_outstanding.empty(),
              "a failed grow must release the old buffer instead of leaking it");

        // Retrying with the allocator working again must succeed instead of
        // treating `bytes` as already covering the request.
        g_fail_on_allocation = 0;
        ensure_buf(gb, 1024);
        check(gb.bytes == 1024, "retrying after a transient failure should grow normally");
    }
    check(g_outstanding.empty(), "leaving scope should release the buffer");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("flux device buffer allocation-failure checks passed\n");
    return 0;
}
