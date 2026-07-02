/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CUDA-CPP-01
// Architecture:   ARCH-MOD-001
// Unit Design:    UD-TRT-CORE-01
// Intent:         CudaBuffer RAII alloc, move semantics, data round-trip
// Preconditions:  CUDA GPU available
// Postconditions: Buffer allocates, moves correctly, data survives H2D->D2H
// =============================================================================

// =============================================================================
// Test suite: CudaBuffer GPU memory wrapper
// =============================================================================
//
// Purpose:
//   Validates the CudaBuffer RAII wrapper from trt_common.h: construction with
//   zero and non-zero sizes, move semantics (constructor and assignment), and
//   data round-trip via cudaMemcpy.
//
// Dependencies:
//   - runtime/core/trt_common.h (CudaBuffer)
//   - CUDA runtime (cudaMalloc, cudaMemcpy)
//
// Environment:
//   Guarded by TRTMC_HAS_TRT. Skips gracefully (exit 0) when TensorRT/CUDA
//   headers are not available. Requires a CUDA-capable GPU at runtime.
// =============================================================================

#include "runtime/core/trt_common.h"

#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// -----------------------------------------------------------------------------
// Intention:  Verify that a zero-size CudaBuffer reports ok()=true with a null
//             data pointer and size 0 — no allocation should occur.
// Setup:      Construct CudaBuffer with size=0.
// Mechanism:  Check ok(), data()==nullptr, size()==0.
// -----------------------------------------------------------------------------
static void test_zero_size_buffer() {
    trtmc::CudaBuffer buf(0);
    check(buf.ok(), "zero_size: ok()=true");
    check(buf.data() == nullptr, "zero_size: data()=nullptr");
    check(buf.size() == 0, "zero_size: size()=0");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that a non-zero CudaBuffer allocates GPU memory and
//             reports correct state.
// Setup:      Construct CudaBuffer with size=1024.
// Mechanism:  Check ok(), data()!=nullptr, size()==1024.
// -----------------------------------------------------------------------------
static void test_nonzero_size_buffer() {
    trtmc::CudaBuffer buf(1024);
    check(buf.ok(), "nonzero_size: ok()=true");
    check(buf.data() != nullptr, "nonzero_size: data()!=nullptr");
    check(buf.size() == 1024, "nonzero_size: size()==1024");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that the move constructor transfers ownership from source
//             to destination, leaving the source in a null/empty state.
// Setup:      Create a CudaBuffer, then move-construct another from it.
// Mechanism:  After move, source should have data()==nullptr and size()==0.
//             Destination should have the original pointer and size.
// -----------------------------------------------------------------------------
static void test_move_constructor() {
    trtmc::CudaBuffer src(512);
    check(src.ok(), "move_ctor: src ok before move");
    void* original_ptr = src.data();

    trtmc::CudaBuffer dst(std::move(src));

    check(dst.ok(), "move_ctor: dst ok after move");
    check(dst.data() == original_ptr, "move_ctor: dst has original ptr");
    check(dst.size() == 512, "move_ctor: dst has original size");
    check(src.data() == nullptr, "move_ctor: src data is nullptr after move");
    check(src.size() == 0, "move_ctor: src size is 0 after move");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that move assignment transfers ownership and properly
//             releases the destination's prior allocation.
// Setup:      Create two CudaBuffers, then move-assign one to the other.
// Mechanism:  After move assignment, source should be null/empty, destination
//             should have the source's original pointer and size.
// -----------------------------------------------------------------------------
static void test_move_assignment() {
    trtmc::CudaBuffer src(256);
    trtmc::CudaBuffer dst(128);
    check(src.ok(), "move_assign: src ok before move");
    check(dst.ok(), "move_assign: dst ok before move");

    void* src_ptr = src.data();

    dst = std::move(src);

    check(dst.ok(), "move_assign: dst ok after move");
    check(dst.data() == src_ptr, "move_assign: dst has src's original ptr");
    check(dst.size() == 256, "move_assign: dst has src's original size");
    check(src.data() == nullptr, "move_assign: src data is nullptr after move");
    check(src.size() == 0, "move_assign: src size is 0 after move");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that data written to a CudaBuffer via cudaMemcpy can be
//             read back correctly — a basic host->device->host round trip.
// Setup:      Create a CudaBuffer, fill it with known float data from the host,
//             then read it back to a separate host buffer.
// Mechanism:  Write {1.0, 2.0, 3.0, 4.0} via cudaMemcpy H2D, then read back
//             via cudaMemcpy D2H and verify values match.
// -----------------------------------------------------------------------------
static void test_data_roundtrip() {
    const std::vector<float> host_data = {1.0F, 2.0F, 3.0F, 4.0F};
    const std::size_t bytes = host_data.size() * sizeof(float);

    trtmc::CudaBuffer buf(bytes);
    check(buf.ok(), "roundtrip: buffer ok");
    check(buf.data() != nullptr, "roundtrip: buffer non-null");

    // H2D
    cudaError_t err = cudaMemcpy(buf.data(), host_data.data(), bytes, cudaMemcpyHostToDevice);
    check(err == cudaSuccess, "roundtrip: cudaMemcpy H2D succeeded");

    // D2H
    std::vector<float> readback(host_data.size(), 0.0F);
    err = cudaMemcpy(readback.data(), buf.data(), bytes, cudaMemcpyDeviceToHost);
    check(err == cudaSuccess, "roundtrip: cudaMemcpy D2H succeeded");

    bool data_matches = true;
    for (std::size_t i = 0; i < host_data.size(); ++i) {
        if (readback[i] != host_data[i]) {
            data_matches = false;
            std::cerr << "roundtrip: mismatch at [" << i << "]: expected " << host_data[i]
                      << " got " << readback[i] << '\n';
        }
    }
    check(data_matches, "roundtrip: data matches after H2D->D2H");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that a large CudaBuffer allocation succeeds and the data
//             round-trips correctly with a pattern fill.
// Setup:      Allocate 1 MB buffer, fill with incrementing bytes, read back.
// Mechanism:  Write 1 MB of data H2D, then read back D2H and compare.
// -----------------------------------------------------------------------------
static void test_large_buffer_roundtrip() {
    const std::size_t num_floats = 256 * 1024; // 1 MB
    const std::size_t bytes = num_floats * sizeof(float);

    trtmc::CudaBuffer buf(bytes);
    check(buf.ok(), "large_roundtrip: buffer ok");

    std::vector<float> host_data(num_floats);
    for (std::size_t i = 0; i < num_floats; ++i) {
        host_data[i] = static_cast<float>(i);
    }

    cudaError_t err = cudaMemcpy(buf.data(), host_data.data(), bytes, cudaMemcpyHostToDevice);
    check(err == cudaSuccess, "large_roundtrip: H2D succeeded");

    std::vector<float> readback(num_floats, 0.0F);
    err = cudaMemcpy(readback.data(), buf.data(), bytes, cudaMemcpyDeviceToHost);
    check(err == cudaSuccess, "large_roundtrip: D2H succeeded");

    bool all_match = true;
    for (std::size_t i = 0; i < num_floats; ++i) {
        if (readback[i] != host_data[i]) {
            all_match = false;
            std::cerr << "large_roundtrip: mismatch at index " << i << ": expected " << host_data[i]
                      << " got " << readback[i] << '\n';
            break;
        }
    }
    check(all_match, "large_roundtrip: all data matches");
}

int main() {
    test_zero_size_buffer();
    test_nonzero_size_buffer();
    test_move_constructor();
    test_move_assignment();
    test_data_roundtrip();
    test_large_buffer_roundtrip();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All CudaBuffer tests passed.\n";
    return 0;
}
