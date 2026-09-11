/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Links the real sampler translation unit against CPU CUDA stubs so each
// allocation can be failed in turn without a GPU. Verifies that a failed
// construction releases every allocation it already made, that a later
// construction still succeeds, and that a failed resize leaves an existing
// sampler usable.

#include "families/bark/runtime/sampler.h"
#include "families/bark/runtime/sparse_multinomial_kernel.h"

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

void reset_allocator(int fail_on) {
    g_fail_on_allocation = fail_on;
    g_allocation_count = 0;
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

cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) {
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t) {
    return cudaSuccess;
}

} // extern "C"

namespace trtmc {

BarkTorchMultinomialExecutionPolicy bark_compute_torch_multinomial_execution_policy(int32_t) {
    BarkTorchMultinomialExecutionPolicy policy;
    policy.total_threads = 32;
    policy.counter_offset = 4;
    return policy;
}

void bark_gpu_sparse_torch_multinomial_exact(const int32_t*, const float*, int32_t, int32_t,
                                             int32_t, uint64_t, uint64_t, int32_t, int32_t*,
                                             cudaStream_t) {}

} // namespace trtmc

int main() {
    // The constructor makes three allocations. Fail each in turn and confirm
    // nothing the call already acquired survives the throw.
    for (int failing = 1; failing <= 3; ++failing) {
        reset_allocator(failing);
        bool threw = false;
        try {
            trtmc::BarkSampler sampler(nullptr);
        } catch (const std::runtime_error& error) {
            threw = true;
            static const char* const expected[] = {
                "Unable to allocate bark sampler index buffer",
                "Unable to allocate bark sampler probability buffer",
                "Unable to allocate bark sampler token id buffer",
            };
            check(std::string(error.what()) == expected[failing - 1],
                  "the error should name the buffer that actually failed");
        }
        check(threw, "construction should throw when an allocation fails");
        check(g_outstanding.empty(), "failed construction must release every allocation it made");
        g_outstanding.clear();
    }

    // A later construction, with no injected failure, still succeeds.
    reset_allocator(0);
    {
        trtmc::BarkSampler sampler(nullptr);
        check(!g_outstanding.empty(), "successful construction should hold its buffers");
    }
    check(g_outstanding.empty(), "destruction should release every buffer");

    // A failed resize of a live sampler keeps the existing buffers intact and
    // leaves the sampler usable.
    reset_allocator(0);
    {
        trtmc::BarkSampler sampler(nullptr);
        const std::set<void*> after_construction = g_outstanding;

        const int32_t vocab_size = 64;
        std::vector<float> logits(static_cast<std::size_t>(vocab_size), 1.0F);

        reset_allocator(1);
        bool threw = false;
        try {
            sampler.sample_rows(logits.data(), 1, vocab_size, vocab_size, 1.0F, vocab_size);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "a failed resize should throw");
        check(g_outstanding == after_construction,
              "a failed resize must leave the existing buffers untouched");

        reset_allocator(0);
        const std::vector<int32_t> tokens =
            sampler.sample_rows(logits.data(), 1, vocab_size, vocab_size, 1.0F, vocab_size);
        check(tokens.size() == 1, "the sampler should still be usable after a failed resize");
    }
    check(g_outstanding.empty(), "destruction should release every buffer");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("bark sampler allocation-failure checks passed\n");
    return 0;
}
