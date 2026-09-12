/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Flux-owned GPU matmul via cuBLAS for preprocessor ops not yet baked into TRT engines.

#include "families/flux/runtime/device_buffer.h"
#include "families/flux/runtime/gpu_matmul.h"

#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace trtmc {

namespace {

cublasHandle_t g_cublas = nullptr;
cudaStream_t g_stream = nullptr;

flux::GrowableBuffer g_dA, g_dB, g_dC;

} // namespace

void flux_gpu_matmul_init() {
    if (g_cublas)
        return;
    cublasCreate(&g_cublas);
    cudaStreamCreate(&g_stream);
    cublasSetStream(g_cublas, g_stream);
}

void flux_gpu_matmul_shutdown() {
    g_dA = flux::GrowableBuffer{};
    g_dB = flux::GrowableBuffer{};
    g_dC = flux::GrowableBuffer{};
    if (g_stream) {
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
    }
    if (g_cublas) {
        cublasDestroy(g_cublas);
        g_cublas = nullptr;
    }
}

void flux_gpu_matmul_bias(const float* A, const float* B, const float* bias, float* out, int32_t M,
                          int32_t K, int32_t N) {
    const size_t sA = size_t(M) * K * sizeof(float);
    const size_t sB = size_t(K) * N * sizeof(float);
    const size_t sC = size_t(M) * N * sizeof(float);

    flux::ensure_buf(g_dA, sA);
    flux::ensure_buf(g_dB, sB);
    flux::ensure_buf(g_dC, sC);

    auto* dA = static_cast<float*>(g_dA.buf.get());
    auto* dB = static_cast<float*>(g_dB.buf.get());
    auto* dC = static_cast<float*>(g_dC.buf.get());

    cudaMemcpyAsync(dA, A, sA, cudaMemcpyHostToDevice, g_stream);
    cudaMemcpyAsync(dB, B, sB, cudaMemcpyHostToDevice, g_stream);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);

    cudaMemcpyAsync(out, dC, sC, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);

    if (bias) {
        for (int32_t i = 0; i < M; ++i)
            for (int32_t j = 0; j < N; ++j)
                out[i * N + j] += bias[j];
    }
}

} // namespace trtmc
