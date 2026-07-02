/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CUDA-CPP-02
// Architecture:   ARCH-MOD-001
// Unit Design:    UD-TRT-CORE-01
// Intent:         CudaStream RAII creation and move semantics
// Preconditions:  CUDA GPU available
// Postconditions: Stream creates and destroys without leak
// =============================================================================

// =============================================================================
// Test suite: CudaStream RAII wrapper
// =============================================================================
//
// Purpose:
//   Validates the CudaStream RAII wrapper from trt_common.h: construction,
//   move semantics (constructor and assignment), and that the underlying
//   cudaStream_t handle is valid.
//
// Dependencies:
//   - runtime/core/trt_common.h (CudaStream)
//   - CUDA runtime (cudaStreamCreate, cudaStreamSynchronize)
//
// Environment:
//   Guarded by TRTMC_HAS_TRT. Skips gracefully (exit 0) when TensorRT/CUDA
//   headers are not available. Requires a CUDA-capable GPU at runtime.
// =============================================================================

#include "runtime/core/trt_common.h"

#include <cuda_runtime_api.h>
#include <iostream>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// -----------------------------------------------------------------------------
// Intention:  Verify that a default-constructed CudaStream creates a valid
//             CUDA stream handle.
// Setup:      Default-construct a CudaStream.
// Mechanism:  Check ok()==true and get()!=nullptr.
// -----------------------------------------------------------------------------
static void test_default_construction() {
    trtmc::CudaStream stream;
    check(stream.ok(), "default_ctor: ok()=true");
    check(stream.get() != nullptr, "default_ctor: get()!=nullptr");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that the CUDA stream handle is usable for synchronization.
// Setup:      Default-construct a CudaStream, then synchronize it.
// Mechanism:  cudaStreamSynchronize should return cudaSuccess on a valid stream.
// -----------------------------------------------------------------------------
static void test_stream_is_usable() {
    trtmc::CudaStream stream;
    check(stream.ok(), "usable: stream ok");
    cudaError_t err = cudaStreamSynchronize(stream.get());
    check(err == cudaSuccess, "usable: cudaStreamSynchronize succeeds");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that the move constructor transfers the stream handle from
//             source to destination, leaving the source with a null handle.
// Setup:      Create a CudaStream, then move-construct another from it.
// Mechanism:  After move, source.get() should be nullptr. Destination should
//             have the original handle.
// -----------------------------------------------------------------------------
static void test_move_constructor() {
    trtmc::CudaStream src;
    check(src.ok(), "move_ctor: src ok before move");
    cudaStream_t original_handle = src.get();

    trtmc::CudaStream dst(std::move(src));

    check(dst.ok(), "move_ctor: dst ok after move");
    check(dst.get() == original_handle, "move_ctor: dst has original handle");
    check(src.get() == nullptr, "move_ctor: src handle is nullptr after move");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that move assignment transfers the stream handle and
//             properly cleans up the destination's prior handle.
// Setup:      Create two CudaStreams, then move-assign one to the other.
// Mechanism:  After move assignment, source should have nullptr handle,
//             destination should have the source's original handle.
// -----------------------------------------------------------------------------
static void test_move_assignment() {
    trtmc::CudaStream src;
    trtmc::CudaStream dst;
    check(src.ok(), "move_assign: src ok before move");
    check(dst.ok(), "move_assign: dst ok before move");

    cudaStream_t src_handle = src.get();

    dst = std::move(src);

    check(dst.ok(), "move_assign: dst ok after move");
    check(dst.get() == src_handle, "move_assign: dst has src's original handle");
    check(src.get() == nullptr, "move_assign: src handle is nullptr after move");
}

// -----------------------------------------------------------------------------
// Intention:  Verify that two independently constructed CudaStreams have
//             distinct handles.
// Setup:      Create two CudaStreams.
// Mechanism:  Their get() pointers should differ.
// -----------------------------------------------------------------------------
static void test_distinct_streams() {
    trtmc::CudaStream a;
    trtmc::CudaStream b;
    check(a.ok(), "distinct: a ok");
    check(b.ok(), "distinct: b ok");
    check(a.get() != b.get(), "distinct: different handles");
}

int main() {
    test_default_construction();
    test_stream_is_usable();
    test_move_constructor();
    test_move_assignment();
    test_distinct_streams();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All CudaStream tests passed.\n";
    return 0;
}
