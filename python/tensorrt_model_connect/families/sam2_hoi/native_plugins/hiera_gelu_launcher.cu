/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Toolchain-independent exact BF16 GELU via the complete raw-bit mapping.
// Oracle: PyTorch e2d141dbde55c2a4370fac5165b0561b6af4798b
// aten/src/ATen/native/cuda/ActivationGeluKernel.cu, compiled by CUDA 12.8.

#include <cstdint>
#include <cuda_runtime_api.h>

namespace sam2_hoi_gelu {

constexpr int kThreads = 128;
constexpr int kElementsPerThread = 8;
constexpr int kElementsPerBlock = kThreads * kElementsPerThread;
constexpr int kVectorSize = 4;
constexpr int kStage0Elements = 1 * 256 * 256 * 384;
constexpr int kStage1Elements = 1 * 128 * 128 * 768;
constexpr int kStage2Elements = 1 * 64 * 64 * 1536;
constexpr int kStage3Elements = 1 * 32 * 32 * 3072;

static_assert(sizeof(std::uint16_t) == 2);

__device__ __align__(128) const std::uint16_t kGeluBf16Lut[1 << 16] = {
#include "hiera_gelu_bf16_lut_cuda128_exact.inc"
};

bool allowed_elements(int n) {
    return n == kStage0Elements || n == kStage1Elements || n == kStage2Elements ||
           n == kStage3Elements;
}

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
    T value[N];
};

struct PointerArray {
    char* data[2]; // output, input
};

__device__ __forceinline__ std::uint16_t gelu_lookup(std::uint16_t input) {
    return __ldg(&kGeluBf16Lut[input]);
}

__global__ __launch_bounds__(kThreads) void gelu_bf16_vectorized_kernel(int n,
                                                                        PointerArray pointers) {
    const int remaining = n - kElementsPerBlock * static_cast<int>(blockIdx.x);
    auto* input =
        reinterpret_cast<const std::uint16_t*>(pointers.data[1]) + kElementsPerBlock * blockIdx.x;
    auto* output =
        reinterpret_cast<std::uint16_t*>(pointers.data[0]) + kElementsPerBlock * blockIdx.x;

    std::uint16_t arguments[kElementsPerThread];
    std::uint16_t results[kElementsPerThread];

    if (remaining < kElementsPerBlock) {
#pragma unroll
        for (int i = 0; i < kElementsPerThread; ++i) {
            const int index = static_cast<int>(threadIdx.x) + i * kThreads;
            if (index < remaining)
                arguments[i] = input[index];
        }
#pragma unroll
        for (int i = 0; i < kElementsPerThread; ++i) {
            const int index = static_cast<int>(threadIdx.x) + i * kThreads;
            if (index < remaining)
                results[i] = gelu_lookup(arguments[i]);
        }
#pragma unroll
        for (int i = 0; i < kElementsPerThread; ++i) {
            const int index = static_cast<int>(threadIdx.x) + i * kThreads;
            if (index < remaining)
                output[index] = results[i];
        }
        return;
    }

    using Vec = AlignedVector<std::uint16_t, kVectorSize>;
    const auto* input_vec = reinterpret_cast<const Vec*>(input);
    auto* output_vec = reinterpret_cast<Vec*>(output);
#pragma unroll
    for (int i = 0; i < kElementsPerThread / kVectorSize; ++i) {
        const Vec loaded = input_vec[threadIdx.x + i * kThreads];
#pragma unroll
        for (int j = 0; j < kVectorSize; ++j) {
            arguments[i * kVectorSize + j] = loaded.value[j];
        }
    }
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
        results[i] = gelu_lookup(arguments[i]);
    }
#pragma unroll
    for (int i = 0; i < kElementsPerThread / kVectorSize; ++i) {
        Vec stored;
#pragma unroll
        for (int j = 0; j < kVectorSize; ++j) {
            stored.value[j] = results[i * kVectorSize + j];
        }
        output_vec[threadIdx.x + i * kThreads] = stored;
    }
}

} // namespace sam2_hoi_gelu

extern "C" int sam2_hoi_hiera_gelu_bf16_shape_allowed(int b, int h, int w, int c) {
    return b == 1 && ((h == 256 && w == 256 && c == 384) || (h == 128 && w == 128 && c == 768) ||
                      (h == 64 && w == 64 && c == 1536) || (h == 32 && w == 32 && c == 3072))
               ? 1
               : 0;
}

extern "C" int sam2_hoi_hiera_gelu_bf16_launch(const void* input, void* output, int n,
                                               cudaStream_t stream) {
    if (input == nullptr || output == nullptr || !sam2_hoi_gelu::allowed_elements(n)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    sam2_hoi_gelu::PointerArray pointers{
        {reinterpret_cast<char*>(output), reinterpret_cast<char*>(const_cast<void*>(input))}};
    const int blocks =
        (n + sam2_hoi_gelu::kElementsPerBlock - 1) / sam2_hoi_gelu::kElementsPerBlock;
    sam2_hoi_gelu::gelu_bf16_vectorized_kernel<<<blocks, sam2_hoi_gelu::kThreads, 0, stream>>>(
        n, pointers);
    return static_cast<int>(cudaPeekAtLastError());
}
