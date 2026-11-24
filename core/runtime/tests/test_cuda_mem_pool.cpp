/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// use_full_size_mem_pool: on an integrated GPU, stream-ordered allocations reach past the
// default pool's limit (a third of the memory on Jetson AGX Thor); elsewhere nothing changes.

#include "runtime/primitives/cuda_common.h"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

namespace {

constexpr int kSkip = 77;

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

cudaMemPool_t current_pool(int device) {
    cudaMemPool_t pool = nullptr;
    cudaDeviceGetMemPool(&pool, device);
    return pool;
}

cudaMemPool_t default_pool(int device) {
    cudaMemPool_t pool = nullptr;
    cudaDeviceGetDefaultMemPool(&pool, device);
    return pool;
}

// Allocate `target` bytes with cudaMallocAsync in eighths of the device memory, then free them.
bool async_allocations_reach(std::size_t target, std::size_t total, cudaStream_t stream) {
    std::vector<void*> held;
    std::size_t reached = 0;
    bool ok = true;
    while (ok && reached < target) {
        void* pointer = nullptr;
        ok = cudaMallocAsync(&pointer, total / 8, stream) == cudaSuccess &&
             cudaStreamSynchronize(stream) == cudaSuccess;
        if (ok) {
            held.push_back(pointer);
            reached += total / 8;
        }
    }
    cudaGetLastError();
    for (void* pointer : held)
        cudaFreeAsync(pointer, stream);
    cudaStreamSynchronize(stream);
    return ok;
}

} // namespace

int main() {
    int device = 0;
    int integrated = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device) != cudaSuccess) {
        std::cout << "SKIP: no CUDA device\n";
        return kSkip;
    }
    if (integrated == 0) {
        const cudaMemPool_t before = current_pool(device);
        check(!trtmc::use_full_size_mem_pool(device), "a discrete GPU keeps its pool");
        check(current_pool(device) == before, "a discrete GPU's current pool is unchanged");
        return failures == 0 ? 0 : 1;
    }

    check(current_pool(device) == default_pool(device), "the process starts on the default pool");
    check(trtmc::use_full_size_mem_pool(device), "an integrated GPU gets a full-size pool");
    check(current_pool(device) != default_pool(device), "the full-size pool is current");
    check(!trtmc::use_full_size_mem_pool(device), "a second call changes nothing");
    check(cudaGetLastError() == cudaSuccess, "no sticky CUDA error is left behind");

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    // Half the memory is past Thor's default limit of a third; it needs that much free memory.
    if (free_bytes < total_bytes / 8 * 5) {
        std::cout << "note: less than 5/8 of the device memory is free; allocation not checked\n";
    } else {
        cudaStream_t stream = nullptr;
        cudaStreamCreate(&stream);
        check(async_allocations_reach(total_bytes / 2, total_bytes, stream),
              "cudaMallocAsync reaches half of the device memory");
        cudaStreamDestroy(stream);
    }
    return failures == 0 ? 0 : 1;
}
