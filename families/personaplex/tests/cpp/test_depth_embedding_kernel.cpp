/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-17
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Depth input embedding rounds at the reference's depth-dtype boundaries
// Preconditions:  A CUDA device (the test is skipped without one)
// Postconditions: An FP16 depth gets Half(Half(projection) + Half(embedding)); an FP32 depth keeps
// the FP32 sum
// =============================================================================

#include "families/personaplex/runtime/sampling_kernels.h"

#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void check_equal(float actual, float expected, const char* name) {
    if (actual != expected) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

template <typename T>
T* upload(const std::vector<T>& values) {
    T* device = nullptr;
    if (cudaMalloc(&device, values.size() * sizeof(T)) != cudaSuccess ||
        cudaMemcpy(device, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice) !=
            cudaSuccess)
        return nullptr;
    return device;
}

// One depth row, temporal hidden [1, 1] in Half, projection row [1, 2^-12] (exact in BF16 and
// Half) and embedding -1 for both the text (codebook 0) and audio (codebook 1) seeds. The
// reference computes F.linear in Half (1 + 2^-12 rounds to 1) and then adds -1 in Half, giving 0;
// a single rounding of the FP32 sum gives 2^-12.
float depth_input(int32_t codebook, trtmc::DType depth_dtype) {
    const std::vector<__half> hidden_values{__float2half(1.0F), __float2half(1.0F)};
    const float tiny = std::ldexp(1.0F, -12);
    const std::vector<float> projection_values{1.0F, tiny, 1.0F, tiny};
    const std::vector<float> embedding_values{-1.0F};
    const std::vector<int32_t> token_values{0, 0};
    auto* hidden = upload(hidden_values);
    auto* projection = upload(projection_values);
    auto* text_embedding = upload(embedding_values);
    auto* audio_embedding = upload(embedding_values);
    auto* tokens = upload(token_values);
    float* output = nullptr;
    float result = NAN;
    if (hidden && projection && text_embedding && audio_embedding && tokens &&
        cudaMalloc(&output, sizeof(float)) == cudaSuccess) {
        trtmc::personaplex_prepare_depth_embedding(output, hidden, trtmc::DType::kFloat16,
                                                   projection, 4, text_embedding, 1,
                                                   audio_embedding, 1, tokens, codebook, 0, false,
                                                   0, false, 1, 2, 1, 1, 1, depth_dtype, nullptr);
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpy(&result, output, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            result = NAN;
    }
    cudaFree(output);
    cudaFree(tokens);
    cudaFree(audio_embedding);
    cudaFree(text_embedding);
    cudaFree(projection);
    cudaFree(hidden);
    return result;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;
    const float tiny = std::ldexp(1.0F, -12);
    check_equal(depth_input(0, trtmc::DType::kFloat16), 0.0F, "half depth text seed");
    check_equal(depth_input(1, trtmc::DType::kFloat16), 0.0F, "half depth audio seed");
    check_equal(depth_input(0, trtmc::DType::kFloat32), tiny, "fp32 depth text seed");
    check_equal(depth_input(1, trtmc::DType::kFloat32), tiny, "fp32 depth audio seed");
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "depth embedding kernel: all checks passed\n";
    return 0;
}
