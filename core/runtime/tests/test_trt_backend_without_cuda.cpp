/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The TensorRT backend must refuse to start when CUDA cannot initialize. TensorRT 11.4's
// createInferRuntime then returns a non-null pointer that crashes on first use, so a backend
// built on it would turn a clear CUDA error into a segmentation fault at model load.

#include "trtmc/runtime/trt_backend.h"

#include <cstdio>
#include <cstdlib>

int main() {
    // Hide every device before anything initializes CUDA in this process.
    if (setenv("CUDA_VISIBLE_DEVICES", "", 1) != 0) {
        std::perror("setenv");
        return 2;
    }
    trtmc::IBackend* backend = trtmc_create_backend();
    if (backend != nullptr) {
        // Do not destroy it: its runtime may be the unusable pointer this test guards against.
        std::fprintf(stderr, "trtmc_create_backend returned a backend without a CUDA device\n");
        return 1;
    }
    return 0;
}
