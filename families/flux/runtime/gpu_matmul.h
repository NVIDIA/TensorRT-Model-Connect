/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Flux-owned GPU matmul via cuBLAS for preprocessor operations
// (context embedder, timestep embedding) that are not yet baked into TRT engines.

#include <cstdint>

namespace trtmc {

void flux_gpu_matmul_init();
void flux_gpu_matmul_shutdown();

// GPU matmul: out[M,N] = A[M,K] @ B[K,N] + bias[N]
void flux_gpu_matmul_bias(const float* A, const float* B, const float* bias, float* out, int32_t M,
                          int32_t K, int32_t N);

} // namespace trtmc
