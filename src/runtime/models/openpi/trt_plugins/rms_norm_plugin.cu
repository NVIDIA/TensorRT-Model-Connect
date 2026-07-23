/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits>
#include <new>
#include <string>

namespace trtmc::openpi {

int32_t launch_openpi_rms_norm(const std::uint16_t* input, const std::uint16_t* scale,
                               std::uint16_t* output, int32_t rows, float epsilon,
                               cudaStream_t stream) noexcept;
int32_t launch_openpi_rope_qk(const std::uint16_t* query, const std::uint16_t* key,
                              const std::int32_t* positions, std::uint16_t* query_output,
                              std::uint16_t* key_output, int32_t sequence, void* workspace,
                              cudaStream_t stream) noexcept;
int32_t launch_openpi_prefix_qk(const std::uint16_t* query, const std::uint16_t* key, float* logits,
                                void* workspace, void* cublas_handle, cudaStream_t stream) noexcept;
int32_t launch_openpi_prefix_softmax(const float* logits, const std::uint8_t* attention_mask,
                                     std::uint16_t* probabilities, cudaStream_t stream) noexcept;
int32_t launch_openpi_final_adaptive_rms_norm(const std::uint16_t* hidden,
                                              const std::uint16_t* bias,
                                              const std::uint16_t* weight, const float* condition,
                                              std::uint16_t* output, float epsilon,
                                              cudaStream_t stream) noexcept;
int32_t launch_openpi_post_attention_rms_norm(
    const std::uint16_t* residual, const std::uint16_t* update, const std::uint16_t* residual_gate,
    const std::uint16_t* scale, const std::uint16_t* shift, std::uint16_t* output, int32_t rows,
    float epsilon, cudaStream_t stream) noexcept;

namespace {

constexpr int32_t kWidth = 2048;
constexpr int32_t kThreads = 256;
constexpr int32_t kRowsPerBlock = 4;
constexpr int32_t kElementsPerThread = 8;
constexpr float kReciprocalWidth = 1.0F / static_cast<float>(kWidth);
constexpr int32_t kRopeSequence = 968;
constexpr int32_t kActionSequence = 15;
constexpr int32_t kRopeHeadDim = 256;
constexpr int32_t kRopeHalf = kRopeHeadDim / 2;
constexpr int32_t kRopeQueryHeads = 8;
constexpr int32_t kRopeKeyHeads = 1;
constexpr std::size_t kRopeTableElements = kRopeSequence * kRopeHalf;
constexpr std::size_t kRopeWorkspaceBytes = (kRopeHalf + 2 * kRopeTableElements) * sizeof(float);
constexpr int32_t kPrefixQKRows = kRopeQueryHeads * kRopeSequence;
constexpr std::size_t kPrefixQKQueryElements =
    static_cast<std::size_t>(kPrefixQKRows) * kRopeHeadDim;
constexpr std::size_t kPrefixQKQueryWorkspaceBytes = kPrefixQKQueryElements * sizeof(std::uint16_t);
// This is the exact scratch allocation handed to cublasGemmEx by the pinned
// JAX 0.5.3 / XLA executable on the qualification GPU.
constexpr std::size_t kPrefixQKCublasWorkspaceBytes = 4460544;
constexpr std::size_t kPrefixQKWorkspaceBytes =
    kPrefixQKQueryWorkspaceBytes + kPrefixQKCublasWorkspaceBytes;
constexpr int32_t kPrefixSoftmaxThreads = 64;
constexpr int32_t kPrefixSoftmaxTiles = 16;
constexpr int32_t kPrefixSoftmaxRows = kRopeQueryHeads * kRopeSequence;
constexpr int32_t kActionWidth = 1024;
constexpr int32_t kActionThreads = 64;
constexpr int32_t kActionElementsPerThread = 16;
constexpr float kActionReciprocalWidth = 1.0F / static_cast<float>(kActionWidth);
constexpr int32_t kFinalAdaptiveRows = 15;
constexpr int32_t kFinalAdaptivePaddedRows = 16;
constexpr int32_t kFinalAdaptiveProjectionWidth = 3 * kActionWidth;
constexpr int32_t kFinalAdaptiveColumnsPerBlock = 16;
constexpr int32_t kFinalAdaptiveBlocks = kActionWidth / kFinalAdaptiveColumnsPerBlock;
constexpr int32_t kFinalAdaptiveThreads = 256;
constexpr int32_t kFinalAdaptiveWarpCount = kFinalAdaptiveThreads / 32;
constexpr int32_t kFinalAdaptiveRmsWarpsPerRow = 4;
constexpr int32_t kFinalAdaptiveConditionTiles = 8;
constexpr float kFinalAdaptiveEpsilon = 1.0e-6F;

__device__ __forceinline__ float bf16_to_float(std::uint16_t value) {
    return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t float_to_bf16_rn(float value) {
    std::uint32_t bits = __float_as_uint(value);
    // Round-to-nearest-even, matching cvt.rn.bf16.f32. Preserve a quiet NaN
    // payload if a non-finite value ever reaches this family-owned primitive.
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U) {
        return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
    }
    bits += 0x00007FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ std::uint16_t bf16_multiply_rn(std::uint16_t lhs, std::uint16_t rhs) {
    return float_to_bf16_rn(__fmul_rn(bf16_to_float(lhs), bf16_to_float(rhs)));
}

__device__ __forceinline__ std::uint16_t bf16_add_rn(std::uint16_t lhs, std::uint16_t rhs) {
    return float_to_bf16_rn(__fadd_rn(bf16_to_float(lhs), bf16_to_float(rhs)));
}

__device__ __forceinline__ float approximate_rsqrt(float value) {
    float result;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int32_t mask = 16; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFFU, value, mask, 32);
    }
    return value;
}

// This is the exact reduction mapping decoded from the pinned JAX/XLA 0.5.3
// GPU kernel. One 256-thread block handles four rows. Every warp owns 256
// contiguous dimensions, every lane accumulates eight contiguous squares,
// the 32 lane totals use a butterfly reduction, and the eight warp totals use
// a second butterfly reduction. XLA then applies CUDA rsqrtf before the two
// separately rounded FP32 products and final BF16 conversion.
__global__ void openpi_rms_norm_kernel(const std::uint16_t* __restrict__ input,
                                       const std::uint16_t* __restrict__ scale,
                                       std::uint16_t* __restrict__ output, int32_t rows,
                                       float epsilon) {
    __shared__ float partials[kRowsPerBlock * 8];

    const int32_t lane = threadIdx.x & 31;
    const int32_t warp = threadIdx.x >> 5;
    const int32_t dimension = threadIdx.x * kElementsPerThread;
    const int32_t first_row = blockIdx.x * kRowsPerBlock;
    float values[kRowsPerBlock][kElementsPerThread];
    float local_sums[kRowsPerBlock]{};

#pragma unroll
    for (int32_t row_offset = 0; row_offset < kRowsPerBlock; ++row_offset) {
        const int32_t row = first_row + row_offset;
        const int64_t base = static_cast<int64_t>(row) * kWidth + dimension;
#pragma unroll
        for (int32_t item = 0; item < kElementsPerThread; ++item) {
            const float value = row < rows ? bf16_to_float(input[base + item]) : 0.0F;
            values[row_offset][item] = value;
            local_sums[row_offset] += value * value;
        }
        const float warp_sum = warp_reduce_sum(local_sums[row_offset]);
        if (lane == 0) {
            partials[row_offset * 8 + warp] = warp_sum;
        }
    }
    __syncthreads();

    if (threadIdx.x < kRowsPerBlock * 8) {
        float total = partials[threadIdx.x];
#pragma unroll
        for (int32_t mask = 4; mask > 0; mask >>= 1) {
            total += __shfl_xor_sync(0xFFFFFFFFU, total, mask, 32);
        }
        if ((threadIdx.x & 7) == 0) {
            partials[threadIdx.x] = total;
        }
    }
    __syncthreads();

    float gamma[kElementsPerThread];
#pragma unroll
    for (int32_t item = 0; item < kElementsPerThread; ++item) {
        // XLA emits fma.rn.bf16(scale, 1, 1). BF16 operands are exactly
        // representable in FP32, so FP32 add followed by explicit BF16 RNE is
        // the same operation and avoids a native-BF16 architecture baseline.
        const float scale_value = bf16_to_float(scale[dimension + item]);
        gamma[item] = bf16_to_float(float_to_bf16_rn(scale_value + 1.0F));
    }

#pragma unroll
    for (int32_t row_offset = 0; row_offset < kRowsPerBlock; ++row_offset) {
        const int32_t row = first_row + row_offset;
        if (row >= rows) {
            continue;
        }
        const float reciprocal = rsqrtf(partials[row_offset * 8] * kReciprocalWidth + epsilon);
        const int64_t base = static_cast<int64_t>(row) * kWidth + dimension;
#pragma unroll
        for (int32_t item = 0; item < kElementsPerThread; ++item) {
            const float normalized = values[row_offset][item] * reciprocal;
            output[base + item] = float_to_bf16_rn(normalized * gamma[item]);
        }
    }
}

__global__ void openpi_rope_period_kernel(float* periods) {
    const int32_t dimension = threadIdx.x;
    if (dimension < kRopeHalf) {
        periods[dimension] = powf(10000.0F, static_cast<float>(dimension) * (2.0F / kRopeHeadDim));
    }
}

__device__ __forceinline__ float full_divide(float numerator, float denominator) {
    float result;
    asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(numerator), "f"(denominator));
    return result;
}

// XLA lowers each position to one 128-thread block. Its CUDA libdevice sine
// and cosine implementations are bit-identical to separate sinf/cosf calls,
// provided the preceding division uses PTX div.full.f32 rather than nvcc's
// usual div.rn.f32 lowering.
__global__ void openpi_rope_table_kernel(const int32_t* __restrict__ positions,
                                         const float* __restrict__ periods,
                                         float* __restrict__ cosine, float* __restrict__ sine) {
    const int32_t position_index = blockIdx.x;
    const int32_t dimension = threadIdx.x;
    const int32_t table_index = position_index * kRopeHalf + dimension;
    const float radians =
        full_divide(static_cast<float>(positions[position_index]), periods[dimension]);
    cosine[table_index] = cosf(radians);
    sine[table_index] = sinf(radians);
}

__global__ void openpi_rope_apply_kernel(const std::uint16_t* __restrict__ input,
                                         const float* __restrict__ cosine,
                                         const float* __restrict__ sine,
                                         std::uint16_t* __restrict__ output, int32_t heads,
                                         int32_t sequence) {
    const int64_t pair_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t pair_count = static_cast<int64_t>(heads) * sequence * kRopeHalf;
    if (pair_index >= pair_count) {
        return;
    }
    const int32_t dimension = pair_index % kRopeHalf;
    const int64_t token_head = pair_index / kRopeHalf;
    const int32_t token = token_head % sequence;
    const int64_t base = token_head * kRopeHeadDim + dimension;
    const int32_t table_index = token * kRopeHalf + dimension;
    const float first = bf16_to_float(input[base]);
    const float second = bf16_to_float(input[base + kRopeHalf]);
    const float cos_value = cosine[table_index];
    const float sin_value = sine[table_index];
    const float out_first = __fsub_rn(__fmul_rn(first, cos_value), __fmul_rn(second, sin_value));
    const float out_second = __fadd_rn(__fmul_rn(second, cos_value), __fmul_rn(first, sin_value));
    output[base] = float_to_bf16_rn(out_first);
    output[base + kRopeHalf] = float_to_bf16_rn(out_second);
}

// TensorRT attention tensors are head-major [B,H,S,D], whereas untouched XLA
// presents one sequence-major [B,S,H,D] matrix to a single m=H*S cuBLAS GEMM.
// Preserve that physical row order: splitting into eight m=S GEMMs selects a
// different reduction kernel and changes FP32 logits despite identical BF16
// operands.
__global__ void openpi_prefix_q_bhsd_to_bshd_kernel(const std::uint16_t* __restrict__ input,
                                                    std::uint16_t* __restrict__ output) {
    const int64_t input_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (input_index >= static_cast<int64_t>(kPrefixQKQueryElements)) {
        return;
    }
    const int32_t dimension = input_index % kRopeHeadDim;
    const int64_t head_token = input_index / kRopeHeadDim;
    const int32_t token = head_token % kRopeSequence;
    const int32_t head = head_token / kRopeSequence;
    const int64_t output_index =
        (static_cast<int64_t>(token) * kRopeQueryHeads + head) * kRopeHeadDim + dimension;
    output[output_index] = input[input_index];
}

// XLA's prefix softmax uses its own exp2 polynomial lowering rather than
// CUDA's __expf implementation. Keep every rounding mode explicit: a change
// in any one of these operations is observable in the BF16 probability cache.
__device__ __forceinline__ float openpi_prefix_expf(float value) {
    const float scaled =
        __fmaf_rn(value, __uint_as_float(0x3BBB989DU), __uint_as_float(0x3F000000U));
    float saturated;
    asm("cvt.sat.f32.f32 %0, %1;" : "=f"(saturated) : "f"(scaled));
    const float exponent =
        __fmaf_rd(saturated, __uint_as_float(0x437C0000U), __uint_as_float(0x4B400001U));
    const float rounded = __fadd_rn(exponent, __uint_as_float(0xCB40007FU));
    float reduced = __fmaf_rn(value, __uint_as_float(0x3FB8AA3BU), -rounded);
    reduced = __fmaf_rn(value, __uint_as_float(0x32A57060U), reduced);
    const float power_of_two = __uint_as_float(__float_as_uint(exponent) << 23U);
    float exponential;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(reduced));
    return __fmul_rn(exponential, power_of_two);
}

// Exact reduction topology decoded from the pinned JAX 0.5.3 / XLA prefix
// program. One 64-thread block owns one head/query row. Each thread processes
// 16 keys separated by 64, then reductions proceed first across eight-lane
// logical groups and finally across the eight group totals. This is deliberately
// family- and shape-specific so unsupported attention shapes fail closed.
__global__ void openpi_prefix_softmax_kernel(const float* __restrict__ logits,
                                             const std::uint8_t* __restrict__ attention_mask,
                                             std::uint16_t* __restrict__ probabilities) {
    const int32_t row = blockIdx.x;
    const int32_t thread = threadIdx.x;
    const int32_t query = row % kRopeSequence;
    constexpr float negative = -2.3819763e38F;
    const int64_t logits_base = static_cast<int64_t>(row) * kRopeSequence;
    const int64_t mask_base = static_cast<int64_t>(query) * kRopeSequence;

    float maximum = __uint_as_float(0xFF800000U);
#pragma unroll
    for (int32_t key_tile = 0; key_tile < kPrefixSoftmaxTiles; ++key_tile) {
        const int32_t key = thread + kPrefixSoftmaxThreads * key_tile;
        if (key < kRopeSequence) {
            const float value =
                attention_mask[mask_base + key] ? logits[logits_base + key] : negative;
            maximum = fmaxf(maximum, value);
        }
    }
#pragma unroll
    for (int32_t offset = 4; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xFFFFFFFFU, maximum, offset, 8));
    }

    __shared__ float partial[8];
    if ((thread & 7) == 0) {
        partial[thread >> 3] = maximum;
    }
    __syncthreads();
    if (thread < 8) {
        maximum = partial[thread];
#pragma unroll
        for (int32_t offset = 4; offset > 0; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0x000000FFU, maximum, offset, 8));
        }
    }
    __syncthreads();
    if (thread == 0) {
        partial[0] = maximum;
    }
    __syncthreads();
    maximum = partial[0];
    // All threads must consume the broadcast maximum before group leaders
    // reuse the same shared array for the exponential-sum reduction.
    __syncthreads();

    float exponentials[kPrefixSoftmaxTiles]{};
    float sum = 0.0F;
#pragma unroll
    for (int32_t key_tile = 0; key_tile < kPrefixSoftmaxTiles; ++key_tile) {
        const int32_t key = thread + kPrefixSoftmaxThreads * key_tile;
        if (key < kRopeSequence) {
            const float value =
                attention_mask[mask_base + key] ? logits[logits_base + key] : negative;
            exponentials[key_tile] = openpi_prefix_expf(__fsub_rn(value, maximum));
            sum = __fadd_rn(sum, exponentials[key_tile]);
        }
    }
#pragma unroll
    for (int32_t offset = 4; offset > 0; offset >>= 1) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xFFFFFFFFU, sum, offset, 8));
    }
    if ((thread & 7) == 0) {
        partial[thread >> 3] = sum;
    }
    __syncthreads();
    if (thread < 8) {
        sum = partial[thread];
#pragma unroll
        for (int32_t offset = 4; offset > 0; offset >>= 1) {
            sum = __fadd_rn(sum, __shfl_xor_sync(0x000000FFU, sum, offset, 8));
        }
    }
    __syncthreads();
    if (thread == 0) {
        partial[0] = sum;
    }
    __syncthreads();
    sum = partial[0];

#pragma unroll
    for (int32_t key_tile = 0; key_tile < kPrefixSoftmaxTiles; ++key_tile) {
        const int32_t key = thread + kPrefixSoftmaxThreads * key_tile;
        if (key < kRopeSequence) {
            probabilities[logits_base + key] =
                float_to_bf16_rn(full_divide(exponentials[key_tile], sum));
        }
    }
}

// Exact final adaptive RMSNorm lowering from the pinned pi05-DROID XLA
// fusion.144. Unlike the per-layer fusion.100 primitive above, one eight-warp
// block owns a 16-channel output tile and jointly reproduces the condition
// projection, all 15 RMS reductions, and the final modulation. Row 15 is the
// zero-padded row present in XLA's [16,1024] tile and is deliberately not
// stored.
__global__ void openpi_final_adaptive_rms_norm_kernel(const std::uint16_t* __restrict__ hidden,
                                                      const std::uint16_t* __restrict__ bias,
                                                      const std::uint16_t* __restrict__ weight,
                                                      const float* __restrict__ condition,
                                                      std::uint16_t* __restrict__ output,
                                                      float epsilon) {
    __shared__ std::uint16_t condition_bf16[kActionWidth];
    __shared__ float rms_partials[kFinalAdaptivePaddedRows][kFinalAdaptiveRmsWarpsPerRow];
    __shared__ float reciprocals[kFinalAdaptivePaddedRows];
    __shared__ float projection_partials[kFinalAdaptiveColumnsPerBlock][kFinalAdaptiveWarpCount];
    __shared__ std::uint16_t modulation[2][kFinalAdaptiveColumnsPerBlock];

    const int32_t thread = threadIdx.x;
    const int32_t lane = thread & 31;
    const int32_t warp = thread >> 5;
    const int32_t column_base = static_cast<int32_t>(blockIdx.x) * kFinalAdaptiveColumnsPerBlock;

    // fusion.144 materializes condition.astype(bfloat16) in 256 four-value
    // vector lanes before either of the two [1024,16] reductions.
    const int32_t condition_base = thread * 4;
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
        const int32_t index = condition_base + item;
        condition_bf16[index] = float_to_bf16_rn(condition[index]);
    }
    __syncthreads();

    // Each row parity uses 128 threads. A thread owns eight contiguous
    // dimensions for every row of that parity; four warp totals are then
    // reduced with the exact XOR(2), XOR(1) tree emitted by Triton.
    const int32_t row_parity = thread >> 7;
    const int32_t first_dimension = (thread & 127) * 8;
#pragma unroll
    for (int32_t row_pair = 0; row_pair < 8; ++row_pair) {
        const int32_t row = row_pair * 2 + row_parity;
        float local_sum = 0.0F;
        if (row < kFinalAdaptiveRows) {
            const int64_t row_base = static_cast<int64_t>(row) * kActionWidth + first_dimension;
            float value = bf16_to_float(hidden[row_base]);
            local_sum = __fmul_rn(value, value);
#pragma unroll
            for (int32_t item = 1; item < 8; ++item) {
                value = bf16_to_float(hidden[row_base + item]);
                local_sum = __fadd_rn(local_sum, __fmul_rn(value, value));
            }
        }
#pragma unroll
        for (int32_t offset = 16; offset > 0; offset >>= 1) {
            local_sum = __fadd_rn(local_sum, __shfl_xor_sync(0xFFFFFFFFU, local_sum, offset, 32));
        }
        if (lane == 0) {
            rms_partials[row][warp & 3] = local_sum;
        }
    }
    __syncthreads();

    if (thread < kFinalAdaptivePaddedRows * kFinalAdaptiveRmsWarpsPerRow) {
        const int32_t row = thread / kFinalAdaptiveRmsWarpsPerRow;
        const int32_t partial = thread & 3;
        float total = rms_partials[row][partial];
        total = __fadd_rn(total, __shfl_xor_sync(0xFFFFFFFFU, total, 2, 32));
        total = __fadd_rn(total, __shfl_xor_sync(0xFFFFFFFFU, total, 1, 32));
        if (partial == 0) {
            const float mean = __fmul_rn(total, kActionReciprocalWidth);
            reciprocals[row] = approximate_rsqrt(__fadd_rn(mean, epsilon));
        }
    }
    __syncthreads();

    // Thread parity selects one eight-column half of this block's output
    // tile. Each thread reduces K={k0,k0+128,...,k0+896} serially. The
    // physical warp combines the 16 lanes of each parity using XOR
    // 16/8/4/2; shared memory then combines all eight warp totals using XOR
    // 4/2/1. This is the reduction order in fusion.144, not a generic GEMM.
    const int32_t column_half = thread & 1;
    const int32_t k0 = ((thread >> 1) & 63) + ((thread >> 7) * 64);
#pragma unroll
    for (int32_t slice = 0; slice < 2; ++slice) {
        const int32_t slice_offset = slice * kActionWidth;
#pragma unroll
        for (int32_t item = 0; item < 8; ++item) {
            const int32_t relative_column = column_half * 8 + item;
            const int32_t output_column = column_base + relative_column;
            std::uint16_t product =
                bf16_multiply_rn(condition_bf16[k0],
                                 weight[static_cast<int64_t>(k0) * kFinalAdaptiveProjectionWidth +
                                        slice_offset + output_column]);
            float local_sum = bf16_to_float(product);
#pragma unroll
            for (int32_t tile = 1; tile < kFinalAdaptiveConditionTiles; ++tile) {
                const int32_t reduction_index =
                    k0 + tile * (kActionWidth / kFinalAdaptiveConditionTiles);
                product = bf16_multiply_rn(
                    condition_bf16[reduction_index],
                    weight[static_cast<int64_t>(reduction_index) * kFinalAdaptiveProjectionWidth +
                           slice_offset + output_column]);
                local_sum = __fadd_rn(local_sum, bf16_to_float(product));
            }
#pragma unroll
            for (int32_t offset = 16; offset >= 2; offset >>= 1) {
                local_sum =
                    __fadd_rn(local_sum, __shfl_xor_sync(0xFFFFFFFFU, local_sum, offset, 32));
            }
            if (lane < 2) {
                projection_partials[relative_column][warp] = local_sum;
            }
        }
        __syncthreads();

        if (thread < kFinalAdaptiveColumnsPerBlock * kFinalAdaptiveWarpCount) {
            const int32_t relative_column = thread / kFinalAdaptiveWarpCount;
            const int32_t partial = thread & (kFinalAdaptiveWarpCount - 1);
            float total = projection_partials[relative_column][partial];
            total = __fadd_rn(total, __shfl_xor_sync(0xFFFFFFFFU, total, 4, 32));
            total = __fadd_rn(total, __shfl_xor_sync(0xFFFFFFFFU, total, 2, 32));
            total = __fadd_rn(total, __shfl_xor_sync(0xFFFFFFFFU, total, 1, 32));
            if (partial == 0) {
                const int32_t parameter_column = slice_offset + column_base + relative_column;
                std::uint16_t value = bf16_add_rn(float_to_bf16_rn(total), bias[parameter_column]);
                if (slice == 0) {
                    value = bf16_add_rn(value, 0x3F80U);
                }
                modulation[slice][relative_column] = value;
            }
        }
        __syncthreads();
    }

    const int32_t row = thread >> 4;
    const int32_t relative_column = thread & 15;
    if (row < kFinalAdaptiveRows) {
        const int64_t index =
            static_cast<int64_t>(row) * kActionWidth + column_base + relative_column;
        const float normalized = __fmul_rn(bf16_to_float(hidden[index]), reciprocals[row]);
        const float scaled = __fmul_rn(normalized, bf16_to_float(modulation[0][relative_column]));
        const float shifted = __fadd_rn(scaled, bf16_to_float(modulation[1][relative_column]));
        output[index] = float_to_bf16_rn(shifted);
    }
}

// Exact arithmetic seam from production fusion.98. The two BF16 roundings in
// the residual path are intentional: XLA emits one native-BF16 FMA for the
// gated update and a second native-BF16 FMA for the residual addition before
// widening the result for the RMS reduction.
__global__ void openpi_post_attention_rms_norm_kernel(
    const std::uint16_t* __restrict__ residual, const std::uint16_t* __restrict__ update,
    const std::uint16_t* __restrict__ residual_gate, const std::uint16_t* __restrict__ scale,
    const std::uint16_t* __restrict__ shift, std::uint16_t* __restrict__ output, int32_t rows,
    float epsilon) {
    __shared__ float partials[2];

    const int32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int32_t lane = threadIdx.x & 31;
    const int32_t warp = threadIdx.x >> 5;
    const int32_t first_dimension = lane * 8 + warp * 256;
    const int64_t row_base = static_cast<int64_t>(row) * kActionWidth;
    float values[kActionElementsPerThread];

#pragma unroll
    for (int32_t item = 0; item < 8; ++item) {
        const int32_t dimensions[2] = {first_dimension + item, first_dimension + 512 + item};
#pragma unroll
        for (int32_t region = 0; region < 2; ++region) {
            const int32_t dimension = dimensions[region];
            const float gated = bf16_to_float(
                float_to_bf16_rn(__fmul_rn(bf16_to_float(update[row_base + dimension]),
                                           bf16_to_float(residual_gate[dimension]))));
            values[item + region * 8] = bf16_to_float(
                float_to_bf16_rn(__fadd_rn(bf16_to_float(residual[row_base + dimension]), gated)));
        }
    }

    float local_sum = __fmul_rn(values[0], values[0]);
#pragma unroll
    for (int32_t item = 1; item < kActionElementsPerThread; ++item) {
        local_sum = __fadd_rn(local_sum, __fmul_rn(values[item], values[item]));
    }
    const float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        partials[warp] = warp_sum;
    }
    __syncthreads();

    float two_warp_value = 0.0F;
    if (threadIdx.x < 2) {
        two_warp_value = partials[threadIdx.x];
    }
    two_warp_value += __shfl_xor_sync(0xFFFFFFFFU, two_warp_value, 1, 32);
    if (threadIdx.x == 0) {
        partials[0] = two_warp_value;
    }
    __syncthreads();

    const float reciprocal =
        rsqrtf(__fadd_rn(__fmul_rn(partials[0], kActionReciprocalWidth), epsilon));
#pragma unroll
    for (int32_t item = 0; item < 8; ++item) {
        const int32_t dimensions[2] = {first_dimension + item, first_dimension + 512 + item};
#pragma unroll
        for (int32_t region = 0; region < 2; ++region) {
            const int32_t dimension = dimensions[region];
            const float gamma =
                bf16_to_float(float_to_bf16_rn(__fadd_rn(bf16_to_float(scale[dimension]), 1.0F)));
            const float normalized = __fmul_rn(values[item + region * 8], reciprocal);
            const float modulated =
                __fadd_rn(__fmul_rn(normalized, gamma), bf16_to_float(shift[dimension]));
            output[row_base + dimension] = float_to_bf16_rn(modulated);
        }
    }
}

bool is_bf16_linear(nvinfer1::PluginTensorDesc const& descriptor) {
    return descriptor.type == nvinfer1::DataType::kBF16 &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR;
}

bool is_float_linear(nvinfer1::PluginTensorDesc const& descriptor) {
    return descriptor.type == nvinfer1::DataType::kFLOAT &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR;
}

bool is_bool_linear(nvinfer1::PluginTensorDesc const& descriptor) {
    return descriptor.type == nvinfer1::DataType::kBOOL &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR;
}

bool has_supported_shape(nvinfer1::Dims const& input, nvinfer1::Dims const& scale) {
    if (input.nbDims < 2 || scale.nbDims != 1 || input.d[input.nbDims - 1] != kWidth ||
        scale.d[0] != kWidth) {
        return false;
    }
    int64_t rows = 1;
    for (int32_t axis = 0; axis + 1 < input.nbDims; ++axis) {
        if (input.d[axis] <= 0) {
            return false;
        }
        rows *= input.d[axis];
    }
    return rows > 0 && rows <= std::numeric_limits<int32_t>::max();
}

bool is_int32_linear(nvinfer1::PluginTensorDesc const& descriptor) {
    return descriptor.type == nvinfer1::DataType::kINT32 &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR;
}

bool has_rope_shape(nvinfer1::Dims const& query, nvinfer1::Dims const& key,
                    nvinfer1::Dims const& positions) {
    if (query.nbDims != 4 || key.nbDims != 4 || positions.nbDims != 2 ||
        (query.d[2] != kRopeSequence && query.d[2] != kActionSequence)) {
        return false;
    }
    const int32_t sequence = query.d[2];
    return query.nbDims == 4 && key.nbDims == 4 && positions.nbDims == 2 && query.d[0] == 1 &&
           query.d[1] == kRopeQueryHeads && query.d[2] == sequence && query.d[3] == kRopeHeadDim &&
           key.d[0] == 1 && key.d[1] == kRopeKeyHeads && key.d[2] == sequence &&
           key.d[3] == kRopeHeadDim && positions.d[0] == 1 && positions.d[1] == sequence;
}

bool has_prefix_qk_shape(nvinfer1::Dims const& query, nvinfer1::Dims const& key) {
    return query.nbDims == 4 && key.nbDims == 4 && query.d[0] == 1 &&
           query.d[1] == kRopeQueryHeads && query.d[2] == kRopeSequence &&
           query.d[3] == kRopeHeadDim && key.d[0] == 1 && key.d[1] == kRopeKeyHeads &&
           key.d[2] == kRopeSequence && key.d[3] == kRopeHeadDim;
}

bool has_prefix_softmax_shape(nvinfer1::Dims const& logits, nvinfer1::Dims const& attention_mask) {
    return logits.nbDims == 4 && attention_mask.nbDims == 4 && logits.d[0] == 1 &&
           logits.d[1] == kRopeQueryHeads && logits.d[2] == kRopeSequence &&
           logits.d[3] == kRopeSequence && attention_mask.d[0] == 1 && attention_mask.d[1] == 1 &&
           attention_mask.d[2] == kRopeSequence && attention_mask.d[3] == kRopeSequence;
}

bool has_supported_adaptive_shape(nvinfer1::Dims const& input, nvinfer1::Dims const& scale,
                                  nvinfer1::Dims const& shift) {
    if (input.nbDims < 2 || scale.nbDims != 1 || shift.nbDims != 1 ||
        input.d[input.nbDims - 1] != kActionWidth || scale.d[0] != kActionWidth ||
        shift.d[0] != kActionWidth) {
        return false;
    }
    int64_t rows = 1;
    for (int32_t axis = 0; axis + 1 < input.nbDims; ++axis) {
        if (input.d[axis] <= 0) {
            return false;
        }
        rows *= input.d[axis];
    }
    return rows > 0 && rows <= std::numeric_limits<int32_t>::max();
}

bool has_final_adaptive_shape(nvinfer1::Dims const& hidden, nvinfer1::Dims const& bias,
                              nvinfer1::Dims const& weight, nvinfer1::Dims const& condition) {
    return hidden.nbDims == 3 && hidden.d[0] == 1 && hidden.d[1] == kFinalAdaptiveRows &&
           hidden.d[2] == kActionWidth && bias.nbDims == 1 &&
           bias.d[0] == kFinalAdaptiveProjectionWidth && weight.nbDims == 2 &&
           weight.d[0] == kActionWidth && weight.d[1] == kFinalAdaptiveProjectionWidth &&
           condition.nbDims == 2 && condition.d[0] == 1 && condition.d[1] == kActionWidth;
}

bool has_final_adaptive_output_shape(nvinfer1::Dims const& output) {
    return output.nbDims == 3 && output.d[0] == 1 && output.d[1] == kFinalAdaptiveRows &&
           output.d[2] == kActionWidth;
}

bool has_supported_post_attention_shape(nvinfer1::Dims const& residual,
                                        nvinfer1::Dims const& update,
                                        nvinfer1::Dims const& residual_gate,
                                        nvinfer1::Dims const& scale, nvinfer1::Dims const& shift) {
    if (!has_supported_adaptive_shape(residual, scale, shift) || update.nbDims != residual.nbDims ||
        residual_gate.nbDims != 1 || residual_gate.d[0] != kActionWidth) {
        return false;
    }
    for (int32_t axis = 0; axis < residual.nbDims; ++axis) {
        if (update.d[axis] != residual.d[axis]) {
            return false;
        }
    }
    return true;
}

class OpenPIRmsNormPlugin final : public nvinfer1::IPluginV3,
                                  public nvinfer1::IPluginV3OneCore,
                                  public nvinfer1::IPluginV3OneBuild,
                                  public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr const char* kName = "OpenPIRmsNorm";
    static constexpr const char* kVersion = "1";

    explicit OpenPIRmsNormPlugin(float epsilon = 1.0e-6F) : epsilon_(epsilon) {
        reset_serialization_fields();
    }

    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override {
        auto* plugin = new (std::nothrow) OpenPIRmsNormPlugin(epsilon_);
        if (plugin != nullptr) {
            plugin->namespace_ = namespace_;
        }
        return plugin;
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* input, int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* output,
                            int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 2 || nb_outputs != 1 ||
            !is_bf16_linear(input[0].desc) || !is_bf16_linear(input[1].desc) ||
            !is_bf16_linear(output[0].desc)) {
            return 1;
        }
        return 0;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t nb_outputs,
                               nvinfer1::DataType const* input_types,
                               int32_t nb_inputs) const noexcept override {
        if (output_types == nullptr || input_types == nullptr || nb_outputs != 1 ||
            nb_inputs != 2 || input_types[0] != nvinfer1::DataType::kBF16 ||
            input_types[1] != nvinfer1::DataType::kBF16) {
            return 1;
        }
        output_types[0] = nvinfer1::DataType::kBF16;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                            nvinfer1::DimsExprs const*, int32_t nb_shape_inputs,
                            nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                            nvinfer1::IExprBuilder&) noexcept override {
        if (inputs == nullptr || outputs == nullptr || nb_inputs != 2 || nb_shape_inputs != 0 ||
            nb_outputs != 1) {
            return 1;
        }
        outputs[0] = inputs[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* descriptors,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        return descriptors != nullptr && nb_inputs == 2 && nb_outputs == 1 && position >= 0 &&
               position < 3 && is_bf16_linear(descriptors[position].desc);
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* input, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* output,
                          int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 2 || nb_outputs != 1 ||
            !has_supported_shape(input[0].dims, input[1].dims) ||
            output[0].dims.nbDims != input[0].dims.nbDims) {
            return 1;
        }
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void*,
                    cudaStream_t stream) noexcept override {
        if (input_desc == nullptr || inputs == nullptr || outputs == nullptr) {
            return 1;
        }
        int64_t rows = 1;
        for (int32_t axis = 0; axis + 1 < input_desc[0].dims.nbDims; ++axis) {
            rows *= input_desc[0].dims.d[axis];
        }
        return launch_openpi_rms_norm(static_cast<const std::uint16_t*>(inputs[0]),
                                      static_cast<const std::uint16_t*>(inputs[1]),
                                      static_cast<std::uint16_t*>(outputs[0]),
                                      static_cast<int32_t>(rows), epsilon_, stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        reset_serialization_fields();
        return &serialization_fields_;
    }

  private:
    void reset_serialization_fields() noexcept {
        serialization_field_ = {"epsilon", &epsilon_, nvinfer1::PluginFieldType::kFLOAT32, 1};
        serialization_fields_.nbFields = 1;
        serialization_fields_.fields = &serialization_field_;
    }

    float epsilon_{1.0e-6F};
    std::string namespace_;
    nvinfer1::PluginField serialization_field_{};
    nvinfer1::PluginFieldCollection serialization_fields_{};
};

class OpenPIRmsNormCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    OpenPIRmsNormCreator() {
        field_ = {"epsilon", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1};
        fields_.nbFields = 1;
        fields_.fields = &field_;
    }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        float epsilon = 1.0e-6F;
        if (fields != nullptr) {
            for (int32_t index = 0; index < fields->nbFields; ++index) {
                const auto& field = fields->fields[index];
                if (field.name != nullptr && std::strcmp(field.name, "epsilon") == 0 &&
                    field.type == nvinfer1::PluginFieldType::kFLOAT32 && field.data != nullptr &&
                    field.length == 1) {
                    epsilon = *static_cast<const float*>(field.data);
                }
            }
        }
        if (!(epsilon > 0.0F)) {
            return nullptr;
        }
        return new (std::nothrow) OpenPIRmsNormPlugin(epsilon);
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return OpenPIRmsNormPlugin::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return OpenPIRmsNormPlugin::kVersion;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginField field_{};
    nvinfer1::PluginFieldCollection fields_{};
};

class OpenPIRopePlugin final : public nvinfer1::IPluginV3,
                               public nvinfer1::IPluginV3OneCore,
                               public nvinfer1::IPluginV3OneBuild,
                               public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr const char* kName = "OpenPIRopeQK";
    static constexpr const char* kVersion = "1";

    OpenPIRopePlugin() { serialization_fields_.nbFields = 0; }

    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override {
        auto* plugin = new (std::nothrow) OpenPIRopePlugin();
        if (plugin != nullptr) {
            plugin->namespace_ = namespace_;
        }
        return plugin;
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* input, int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* output,
                            int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 3 || nb_outputs != 2 ||
            !is_bf16_linear(input[0].desc) || !is_bf16_linear(input[1].desc) ||
            !is_int32_linear(input[2].desc) || !is_bf16_linear(output[0].desc) ||
            !is_bf16_linear(output[1].desc)) {
            return 1;
        }
        return 0;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t nb_outputs,
                               nvinfer1::DataType const* input_types,
                               int32_t nb_inputs) const noexcept override {
        if (output_types == nullptr || input_types == nullptr || nb_outputs != 2 ||
            nb_inputs != 3 || input_types[0] != nvinfer1::DataType::kBF16 ||
            input_types[1] != nvinfer1::DataType::kBF16 ||
            input_types[2] != nvinfer1::DataType::kINT32) {
            return 1;
        }
        output_types[0] = nvinfer1::DataType::kBF16;
        output_types[1] = nvinfer1::DataType::kBF16;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                            nvinfer1::DimsExprs const*, int32_t nb_shape_inputs,
                            nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                            nvinfer1::IExprBuilder&) noexcept override {
        if (inputs == nullptr || outputs == nullptr || nb_inputs != 3 || nb_shape_inputs != 0 ||
            nb_outputs != 2) {
            return 1;
        }
        outputs[0] = inputs[0];
        outputs[1] = inputs[1];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* descriptors,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        if (descriptors == nullptr || nb_inputs != 3 || nb_outputs != 2 || position < 0 ||
            position >= 5) {
            return false;
        }
        return position == 2 ? is_int32_linear(descriptors[position].desc)
                             : is_bf16_linear(descriptors[position].desc);
    }

    int32_t getNbOutputs() const noexcept override { return 2; }

    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                 nvinfer1::DynamicPluginTensorDesc const*,
                                 int32_t) const noexcept override {
        return kRopeWorkspaceBytes;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* input, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* output,
                          int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 3 || nb_outputs != 2 ||
            !has_rope_shape(input[0].dims, input[1].dims, input[2].dims) ||
            output[0].dims.nbDims != 4 || output[1].dims.nbDims != 4) {
            return 1;
        }
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        if (input_desc == nullptr || inputs == nullptr || outputs == nullptr ||
            workspace == nullptr) {
            return 1;
        }
        return launch_openpi_rope_qk(
            static_cast<const std::uint16_t*>(inputs[0]),
            static_cast<const std::uint16_t*>(inputs[1]),
            static_cast<const std::int32_t*>(inputs[2]), static_cast<std::uint16_t*>(outputs[0]),
            static_cast<std::uint16_t*>(outputs[1]), input_desc[0].dims.d[2], workspace, stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        return &serialization_fields_;
    }

  private:
    std::string namespace_;
    nvinfer1::PluginFieldCollection serialization_fields_{};
};

class OpenPIRopeCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    OpenPIRopeCreator() { fields_.nbFields = 0; }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const*,
                                      nvinfer1::TensorRTPhase) noexcept override {
        return new (std::nothrow) OpenPIRopePlugin();
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return OpenPIRopePlugin::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return OpenPIRopePlugin::kVersion;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginFieldCollection fields_{};
};

class OpenPIPrefixQKPlugin final : public nvinfer1::IPluginV3,
                                   public nvinfer1::IPluginV3OneCore,
                                   public nvinfer1::IPluginV3OneBuild,
                                   public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr const char* kName = "OpenPIPrefixQK";
    static constexpr const char* kVersion = "1";

    OpenPIPrefixQKPlugin() { serialization_fields_.nbFields = 0; }

    ~OpenPIPrefixQKPlugin() override {
        if (cublas_handle_ != nullptr) {
            const cublasStatus_t status = cublasDestroy(cublas_handle_);
            // Destructors cannot report an error through TensorRT. Still
            // evaluate the return value so no cuBLAS call is unchecked.
            if (status != CUBLAS_STATUS_SUCCESS) {
                cublas_handle_ = nullptr;
                return;
            }
            cublas_handle_ = nullptr;
        }
    }

    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override {
        auto* plugin = new (std::nothrow) OpenPIPrefixQKPlugin();
        if (plugin != nullptr) {
            plugin->namespace_ = namespace_;
        }
        return plugin;
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* input, int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* output,
                            int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 2 || nb_outputs != 1 ||
            !is_bf16_linear(input[0].desc) || !is_bf16_linear(input[1].desc) ||
            !is_float_linear(output[0].desc)) {
            return 1;
        }
        return 0;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t nb_outputs,
                               nvinfer1::DataType const* input_types,
                               int32_t nb_inputs) const noexcept override {
        if (output_types == nullptr || input_types == nullptr || nb_outputs != 1 ||
            nb_inputs != 2 || input_types[0] != nvinfer1::DataType::kBF16 ||
            input_types[1] != nvinfer1::DataType::kBF16) {
            return 1;
        }
        output_types[0] = nvinfer1::DataType::kFLOAT;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                            nvinfer1::DimsExprs const*, int32_t nb_shape_inputs,
                            nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                            nvinfer1::IExprBuilder&) noexcept override {
        if (inputs == nullptr || outputs == nullptr || nb_inputs != 2 || nb_shape_inputs != 0 ||
            nb_outputs != 1) {
            return 1;
        }
        // Raw cublasGemmEx C is column-major [H*S,S]. TensorRT views the same
        // bytes row-major as [S,S,H], after which one shuffle transposes it to
        // the public [H,S,S] attention layout without another arithmetic seam.
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[0].d[2];
        outputs[0].d[2] = inputs[0].d[2];
        outputs[0].d[3] = inputs[0].d[1];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* descriptors,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        if (descriptors == nullptr || nb_inputs != 2 || nb_outputs != 1 || position < 0 ||
            position >= 3) {
            return false;
        }
        return position < 2 ? is_bf16_linear(descriptors[position].desc)
                            : is_float_linear(descriptors[position].desc);
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                 nvinfer1::DynamicPluginTensorDesc const*,
                                 int32_t) const noexcept override {
        return kPrefixQKWorkspaceBytes;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* input, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* output,
                          int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 2 || nb_outputs != 1 ||
            !has_prefix_qk_shape(input[0].dims, input[1].dims) || output[0].dims.nbDims != 4 ||
            output[0].dims.d[0] != 1 || output[0].dims.d[1] != kRopeSequence ||
            output[0].dims.d[2] != kRopeSequence || output[0].dims.d[3] != kRopeQueryHeads) {
            return 1;
        }
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        if (inputs == nullptr || outputs == nullptr || workspace == nullptr ||
            cublas_handle_ == nullptr) {
            return 1;
        }
        return launch_openpi_prefix_qk(static_cast<const std::uint16_t*>(inputs[0]),
                                       static_cast<const std::uint16_t*>(inputs[1]),
                                       static_cast<float*>(outputs[0]), workspace,
                                       static_cast<void*>(cublas_handle_), stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        auto* plugin = new (std::nothrow) OpenPIPrefixQKPlugin();
        if (plugin == nullptr) {
            return nullptr;
        }
        plugin->namespace_ = namespace_;
        cublasStatus_t status = cublasCreate(&plugin->cublas_handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            delete plugin;
            return nullptr;
        }
        status = cublasSetMathMode(plugin->cublas_handle_, CUBLAS_DEFAULT_MATH);
        if (status != CUBLAS_STATUS_SUCCESS) {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        return &serialization_fields_;
    }

  private:
    std::string namespace_;
    nvinfer1::PluginFieldCollection serialization_fields_{};
    cublasHandle_t cublas_handle_{nullptr};
};

class OpenPIPrefixQKCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    OpenPIPrefixQKCreator() { fields_.nbFields = 0; }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const*,
                                      nvinfer1::TensorRTPhase) noexcept override {
        return new (std::nothrow) OpenPIPrefixQKPlugin();
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return OpenPIPrefixQKPlugin::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return OpenPIPrefixQKPlugin::kVersion;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginFieldCollection fields_{};
};

class OpenPIPrefixSoftmaxPlugin final : public nvinfer1::IPluginV3,
                                        public nvinfer1::IPluginV3OneCore,
                                        public nvinfer1::IPluginV3OneBuild,
                                        public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr const char* kName = "OpenPIPrefixSoftmax";
    static constexpr const char* kVersion = "1";

    OpenPIPrefixSoftmaxPlugin() { serialization_fields_.nbFields = 0; }

    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override {
        auto* plugin = new (std::nothrow) OpenPIPrefixSoftmaxPlugin();
        if (plugin != nullptr) {
            plugin->namespace_ = namespace_;
        }
        return plugin;
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* input, int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* output,
                            int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 2 || nb_outputs != 1 ||
            !is_float_linear(input[0].desc) || !is_bool_linear(input[1].desc) ||
            !is_bf16_linear(output[0].desc)) {
            return 1;
        }
        return 0;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t nb_outputs,
                               nvinfer1::DataType const* input_types,
                               int32_t nb_inputs) const noexcept override {
        if (output_types == nullptr || input_types == nullptr || nb_outputs != 1 ||
            nb_inputs != 2 || input_types[0] != nvinfer1::DataType::kFLOAT ||
            input_types[1] != nvinfer1::DataType::kBOOL) {
            return 1;
        }
        output_types[0] = nvinfer1::DataType::kBF16;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                            nvinfer1::DimsExprs const*, int32_t nb_shape_inputs,
                            nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                            nvinfer1::IExprBuilder&) noexcept override {
        if (inputs == nullptr || outputs == nullptr || nb_inputs != 2 || nb_shape_inputs != 0 ||
            nb_outputs != 1) {
            return 1;
        }
        outputs[0] = inputs[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* descriptors,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        if (descriptors == nullptr || nb_inputs != 2 || nb_outputs != 1 || position < 0 ||
            position >= 3) {
            return false;
        }
        if (position == 0) {
            return is_float_linear(descriptors[position].desc);
        }
        if (position == 1) {
            return is_bool_linear(descriptors[position].desc);
        }
        return is_bf16_linear(descriptors[position].desc);
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                 nvinfer1::DynamicPluginTensorDesc const*,
                                 int32_t) const noexcept override {
        return 0;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* input, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* output,
                          int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 2 || nb_outputs != 1 ||
            !has_prefix_softmax_shape(input[0].dims, input[1].dims) || output[0].dims.nbDims != 4 ||
            output[0].dims.d[0] != 1 || output[0].dims.d[1] != kRopeQueryHeads ||
            output[0].dims.d[2] != kRopeSequence || output[0].dims.d[3] != kRopeSequence) {
            return 1;
        }
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void*,
                    cudaStream_t stream) noexcept override {
        if (inputs == nullptr || outputs == nullptr) {
            return 1;
        }
        return launch_openpi_prefix_softmax(static_cast<const float*>(inputs[0]),
                                            static_cast<const std::uint8_t*>(inputs[1]),
                                            static_cast<std::uint16_t*>(outputs[0]), stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        return &serialization_fields_;
    }

  private:
    std::string namespace_;
    nvinfer1::PluginFieldCollection serialization_fields_{};
};

class OpenPIPrefixSoftmaxCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    OpenPIPrefixSoftmaxCreator() { fields_.nbFields = 0; }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const*,
                                      nvinfer1::TensorRTPhase) noexcept override {
        return new (std::nothrow) OpenPIPrefixSoftmaxPlugin();
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return OpenPIPrefixSoftmaxPlugin::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return OpenPIPrefixSoftmaxPlugin::kVersion;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginFieldCollection fields_{};
};

class OpenPIFinalAdaptiveRmsNormPlugin final : public nvinfer1::IPluginV3,
                                               public nvinfer1::IPluginV3OneCore,
                                               public nvinfer1::IPluginV3OneBuild,
                                               public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr const char* kName = "OpenPIFinalAdaptiveRmsNorm";
    static constexpr const char* kVersion = "1";

    explicit OpenPIFinalAdaptiveRmsNormPlugin(float epsilon = kFinalAdaptiveEpsilon)
        : epsilon_(epsilon) {
        reset_serialization_fields();
    }

    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override {
        auto* plugin = new (std::nothrow) OpenPIFinalAdaptiveRmsNormPlugin(epsilon_);
        if (plugin != nullptr) {
            plugin->namespace_ = namespace_;
        }
        return plugin;
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* input, int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* output,
                            int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 4 || nb_outputs != 1 ||
            !is_bf16_linear(input[0].desc) || !is_bf16_linear(input[1].desc) ||
            !is_bf16_linear(input[2].desc) || !is_float_linear(input[3].desc) ||
            !is_bf16_linear(output[0].desc) ||
            !has_final_adaptive_shape(input[0].desc.dims, input[1].desc.dims, input[2].desc.dims,
                                      input[3].desc.dims) ||
            !has_final_adaptive_shape(input[0].min, input[1].min, input[2].min, input[3].min) ||
            !has_final_adaptive_shape(input[0].max, input[1].max, input[2].max, input[3].max) ||
            !has_final_adaptive_output_shape(output[0].desc.dims) ||
            !has_final_adaptive_output_shape(output[0].min) ||
            !has_final_adaptive_output_shape(output[0].max)) {
            return 1;
        }
        return 0;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t nb_outputs,
                               nvinfer1::DataType const* input_types,
                               int32_t nb_inputs) const noexcept override {
        if (output_types == nullptr || input_types == nullptr || nb_outputs != 1 ||
            nb_inputs != 4 || input_types[0] != nvinfer1::DataType::kBF16 ||
            input_types[1] != nvinfer1::DataType::kBF16 ||
            input_types[2] != nvinfer1::DataType::kBF16 ||
            input_types[3] != nvinfer1::DataType::kFLOAT) {
            return 1;
        }
        output_types[0] = nvinfer1::DataType::kBF16;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                            nvinfer1::DimsExprs const*, int32_t nb_shape_inputs,
                            nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                            nvinfer1::IExprBuilder&) noexcept override {
        if (inputs == nullptr || outputs == nullptr || nb_inputs != 4 || nb_shape_inputs != 0 ||
            nb_outputs != 1) {
            return 1;
        }
        outputs[0] = inputs[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* descriptors,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        if (descriptors == nullptr || nb_inputs != 4 || nb_outputs != 1 || position < 0 ||
            position >= 5) {
            return false;
        }
        return position == 3 ? is_float_linear(descriptors[position].desc)
                             : is_bf16_linear(descriptors[position].desc);
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                 nvinfer1::DynamicPluginTensorDesc const*,
                                 int32_t) const noexcept override {
        return 0;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* input, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* output,
                          int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 4 || nb_outputs != 1 ||
            !is_bf16_linear(input[0]) || !is_bf16_linear(input[1]) || !is_bf16_linear(input[2]) ||
            !is_float_linear(input[3]) || !is_bf16_linear(output[0]) ||
            !has_final_adaptive_shape(input[0].dims, input[1].dims, input[2].dims, input[3].dims) ||
            !has_final_adaptive_output_shape(output[0].dims)) {
            return 1;
        }
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc, void const* const* inputs,
                    void* const* outputs, void*, cudaStream_t stream) noexcept override {
        if (input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
            outputs == nullptr || !is_bf16_linear(input_desc[0]) ||
            !is_bf16_linear(input_desc[1]) || !is_bf16_linear(input_desc[2]) ||
            !is_float_linear(input_desc[3]) || !is_bf16_linear(output_desc[0]) ||
            !has_final_adaptive_shape(input_desc[0].dims, input_desc[1].dims, input_desc[2].dims,
                                      input_desc[3].dims) ||
            !has_final_adaptive_output_shape(output_desc[0].dims)) {
            return 1;
        }
        return launch_openpi_final_adaptive_rms_norm(
            static_cast<const std::uint16_t*>(inputs[0]),
            static_cast<const std::uint16_t*>(inputs[1]),
            static_cast<const std::uint16_t*>(inputs[2]), static_cast<const float*>(inputs[3]),
            static_cast<std::uint16_t*>(outputs[0]), epsilon_, stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        reset_serialization_fields();
        return &serialization_fields_;
    }

  private:
    void reset_serialization_fields() noexcept {
        serialization_field_ = {"epsilon", &epsilon_, nvinfer1::PluginFieldType::kFLOAT32, 1};
        serialization_fields_.nbFields = 1;
        serialization_fields_.fields = &serialization_field_;
    }

    float epsilon_{kFinalAdaptiveEpsilon};
    std::string namespace_;
    nvinfer1::PluginField serialization_field_{};
    nvinfer1::PluginFieldCollection serialization_fields_{};
};

class OpenPIFinalAdaptiveRmsNormCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    OpenPIFinalAdaptiveRmsNormCreator() {
        field_ = {"epsilon", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1};
        fields_.nbFields = 1;
        fields_.fields = &field_;
    }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields == nullptr || fields->fields == nullptr || fields->nbFields != 1) {
            return nullptr;
        }
        const auto& field = fields->fields[0];
        if (field.name == nullptr || std::strcmp(field.name, "epsilon") != 0 ||
            field.type != nvinfer1::PluginFieldType::kFLOAT32 || field.data == nullptr ||
            field.length != 1) {
            return nullptr;
        }
        const float epsilon = *static_cast<const float*>(field.data);
        if (epsilon != kFinalAdaptiveEpsilon) {
            return nullptr;
        }
        return new (std::nothrow) OpenPIFinalAdaptiveRmsNormPlugin(epsilon);
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return OpenPIFinalAdaptiveRmsNormPlugin::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return OpenPIFinalAdaptiveRmsNormPlugin::kVersion;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginField field_{};
    nvinfer1::PluginFieldCollection fields_{};
};

class OpenPIPostAttentionRmsNormPlugin final : public nvinfer1::IPluginV3,
                                               public nvinfer1::IPluginV3OneCore,
                                               public nvinfer1::IPluginV3OneBuild,
                                               public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr const char* kName = "OpenPIPostAttentionRmsNorm";
    static constexpr const char* kVersion = "1";

    explicit OpenPIPostAttentionRmsNormPlugin(float epsilon = 1.0e-6F) : epsilon_(epsilon) {
        reset_serialization_fields();
    }

    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override {
        auto* plugin = new (std::nothrow) OpenPIPostAttentionRmsNormPlugin(epsilon_);
        if (plugin != nullptr) {
            plugin->namespace_ = namespace_;
        }
        return plugin;
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* input, int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* output,
                            int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 5 || nb_outputs != 1 ||
            !is_bf16_linear(input[0].desc) || !is_bf16_linear(input[1].desc) ||
            !is_bf16_linear(input[2].desc) || !is_bf16_linear(input[3].desc) ||
            !is_bf16_linear(input[4].desc) || !is_bf16_linear(output[0].desc)) {
            return 1;
        }
        return 0;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t nb_outputs,
                               nvinfer1::DataType const* input_types,
                               int32_t nb_inputs) const noexcept override {
        if (output_types == nullptr || input_types == nullptr || nb_outputs != 1 ||
            nb_inputs != 5) {
            return 1;
        }
        for (int32_t index = 0; index < nb_inputs; ++index) {
            if (input_types[index] != nvinfer1::DataType::kBF16) {
                return 1;
            }
        }
        output_types[0] = nvinfer1::DataType::kBF16;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                            nvinfer1::DimsExprs const*, int32_t nb_shape_inputs,
                            nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                            nvinfer1::IExprBuilder&) noexcept override {
        if (inputs == nullptr || outputs == nullptr || nb_inputs != 5 || nb_shape_inputs != 0 ||
            nb_outputs != 1) {
            return 1;
        }
        outputs[0] = inputs[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* descriptors,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        return descriptors != nullptr && nb_inputs == 5 && nb_outputs == 1 && position >= 0 &&
               position < 6 && is_bf16_linear(descriptors[position].desc);
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* input, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* output,
                          int32_t nb_outputs) noexcept override {
        if (input == nullptr || output == nullptr || nb_inputs != 5 || nb_outputs != 1 ||
            !has_supported_post_attention_shape(input[0].dims, input[1].dims, input[2].dims,
                                                input[3].dims, input[4].dims) ||
            output[0].dims.nbDims != input[0].dims.nbDims) {
            return 1;
        }
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void*,
                    cudaStream_t stream) noexcept override {
        if (input_desc == nullptr || inputs == nullptr || outputs == nullptr) {
            return 1;
        }
        int64_t rows = 1;
        for (int32_t axis = 0; axis + 1 < input_desc[0].dims.nbDims; ++axis) {
            rows *= input_desc[0].dims.d[axis];
        }
        return launch_openpi_post_attention_rms_norm(static_cast<const std::uint16_t*>(inputs[0]),
                                                     static_cast<const std::uint16_t*>(inputs[1]),
                                                     static_cast<const std::uint16_t*>(inputs[2]),
                                                     static_cast<const std::uint16_t*>(inputs[3]),
                                                     static_cast<const std::uint16_t*>(inputs[4]),
                                                     static_cast<std::uint16_t*>(outputs[0]),
                                                     static_cast<int32_t>(rows), epsilon_, stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        reset_serialization_fields();
        return &serialization_fields_;
    }

  private:
    void reset_serialization_fields() noexcept {
        serialization_field_ = {"epsilon", &epsilon_, nvinfer1::PluginFieldType::kFLOAT32, 1};
        serialization_fields_.nbFields = 1;
        serialization_fields_.fields = &serialization_field_;
    }

    float epsilon_{1.0e-6F};
    std::string namespace_;
    nvinfer1::PluginField serialization_field_{};
    nvinfer1::PluginFieldCollection serialization_fields_{};
};

class OpenPIPostAttentionRmsNormCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    OpenPIPostAttentionRmsNormCreator() {
        field_ = {"epsilon", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1};
        fields_.nbFields = 1;
        fields_.fields = &field_;
    }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        float epsilon = 1.0e-6F;
        if (fields != nullptr) {
            for (int32_t index = 0; index < fields->nbFields; ++index) {
                const auto& field = fields->fields[index];
                if (field.name != nullptr && std::strcmp(field.name, "epsilon") == 0 &&
                    field.type == nvinfer1::PluginFieldType::kFLOAT32 && field.data != nullptr &&
                    field.length == 1) {
                    epsilon = *static_cast<const float*>(field.data);
                }
            }
        }
        if (!(epsilon > 0.0F)) {
            return nullptr;
        }
        return new (std::nothrow) OpenPIPostAttentionRmsNormPlugin(epsilon);
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return OpenPIPostAttentionRmsNormPlugin::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return OpenPIPostAttentionRmsNormPlugin::kVersion;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginField field_{};
    nvinfer1::PluginFieldCollection fields_{};
};

} // namespace

int32_t launch_openpi_rms_norm(const std::uint16_t* input, const std::uint16_t* scale,
                               std::uint16_t* output, int32_t rows, float epsilon,
                               cudaStream_t stream) noexcept {
    if (input == nullptr || scale == nullptr || output == nullptr || rows <= 0 ||
        !(epsilon > 0.0F)) {
        return 1;
    }
    const int32_t blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    openpi_rms_norm_kernel<<<blocks, kThreads, 0, stream>>>(input, scale, output, rows, epsilon);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

int32_t launch_openpi_rope_qk(const std::uint16_t* query, const std::uint16_t* key,
                              const std::int32_t* positions, std::uint16_t* query_output,
                              std::uint16_t* key_output, int32_t sequence, void* workspace,
                              cudaStream_t stream) noexcept {
    if (query == nullptr || key == nullptr || positions == nullptr || query_output == nullptr ||
        key_output == nullptr || workspace == nullptr ||
        (sequence != kRopeSequence && sequence != kActionSequence)) {
        return 1;
    }
    auto* periods = static_cast<float*>(workspace);
    auto* cosine = periods + kRopeHalf;
    auto* sine = cosine + kRopeTableElements;
    openpi_rope_period_kernel<<<1, kRopeHalf, 0, stream>>>(periods);
    openpi_rope_table_kernel<<<sequence, kRopeHalf, 0, stream>>>(positions, periods, cosine, sine);
    constexpr int32_t threads = 256;
    const int32_t query_pairs = kRopeQueryHeads * sequence * kRopeHalf;
    const int32_t key_pairs = kRopeKeyHeads * sequence * kRopeHalf;
    openpi_rope_apply_kernel<<<(query_pairs + threads - 1) / threads, threads, 0, stream>>>(
        query, cosine, sine, query_output, kRopeQueryHeads, sequence);
    openpi_rope_apply_kernel<<<(key_pairs + threads - 1) / threads, threads, 0, stream>>>(
        key, cosine, sine, key_output, kRopeKeyHeads, sequence);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

int32_t launch_openpi_prefix_qk(const std::uint16_t* query, const std::uint16_t* key, float* logits,
                                void* workspace, void* cublas_handle,
                                cudaStream_t stream) noexcept {
    if (query == nullptr || key == nullptr || logits == nullptr || workspace == nullptr ||
        cublas_handle == nullptr) {
        return 1;
    }

    auto* query_sequence_major = static_cast<std::uint16_t*>(workspace);
    auto* cublas_workspace =
        static_cast<void*>(static_cast<std::byte*>(workspace) + kPrefixQKQueryWorkspaceBytes);
    constexpr int32_t threads = 256;
    constexpr int32_t blocks =
        static_cast<int32_t>((kPrefixQKQueryElements + threads - 1) / threads);
    openpi_prefix_q_bhsd_to_bshd_kernel<<<blocks, threads, 0, stream>>>(query,
                                                                        query_sequence_major);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return 1;
    }

    auto handle = reinterpret_cast<cublasHandle_t>(cublas_handle);
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }
    status = cublasSetWorkspace(handle, cublas_workspace, kPrefixQKCublasWorkspaceBytes);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }

    const float alpha = 1.0F;
    const float beta = 0.0F;
    status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kPrefixQKRows, kRopeSequence,
                          kRopeHeadDim, &alpha, query_sequence_major, CUDA_R_16BF, kRopeHeadDim,
                          key, CUDA_R_16BF, kRopeHeadDim, &beta, logits, CUDA_R_32F, kPrefixQKRows,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

int32_t launch_openpi_prefix_softmax(const float* logits, const std::uint8_t* attention_mask,
                                     std::uint16_t* probabilities, cudaStream_t stream) noexcept {
    if (logits == nullptr || attention_mask == nullptr || probabilities == nullptr) {
        return 1;
    }
    openpi_prefix_softmax_kernel<<<kPrefixSoftmaxRows, kPrefixSoftmaxThreads, 0, stream>>>(
        logits, attention_mask, probabilities);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

int32_t launch_openpi_final_adaptive_rms_norm(const std::uint16_t* hidden,
                                              const std::uint16_t* bias,
                                              const std::uint16_t* weight, const float* condition,
                                              std::uint16_t* output, float epsilon,
                                              cudaStream_t stream) noexcept {
    if (hidden == nullptr || bias == nullptr || weight == nullptr || condition == nullptr ||
        output == nullptr || epsilon != kFinalAdaptiveEpsilon) {
        return 1;
    }
    openpi_final_adaptive_rms_norm_kernel<<<kFinalAdaptiveBlocks, kFinalAdaptiveThreads, 0,
                                            stream>>>(hidden, bias, weight, condition, output,
                                                      epsilon);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

int32_t launch_openpi_post_attention_rms_norm(
    const std::uint16_t* residual, const std::uint16_t* update, const std::uint16_t* residual_gate,
    const std::uint16_t* scale, const std::uint16_t* shift, std::uint16_t* output, int32_t rows,
    float epsilon, cudaStream_t stream) noexcept {
    if (residual == nullptr || update == nullptr || residual_gate == nullptr || scale == nullptr ||
        shift == nullptr || output == nullptr || rows <= 0 || !(epsilon > 0.0F)) {
        return 1;
    }
    openpi_post_attention_rms_norm_kernel<<<rows, kActionThreads, 0, stream>>>(
        residual, update, residual_gate, scale, shift, output, rows, epsilon);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace trtmc::openpi

static nvinfer1::PluginRegistrar<trtmc::openpi::OpenPIRmsNormCreator>
    plugin_registrar_openpi_rms_norm{};
static nvinfer1::PluginRegistrar<trtmc::openpi::OpenPIRopeCreator> plugin_registrar_openpi_rope{};
static nvinfer1::PluginRegistrar<trtmc::openpi::OpenPIPrefixQKCreator>
    plugin_registrar_openpi_prefix_qk{};
static nvinfer1::PluginRegistrar<trtmc::openpi::OpenPIPrefixSoftmaxCreator>
    plugin_registrar_openpi_prefix_softmax{};
static nvinfer1::PluginRegistrar<trtmc::openpi::OpenPIFinalAdaptiveRmsNormCreator>
    plugin_registrar_openpi_final_adaptive_rms_norm{};
static nvinfer1::PluginRegistrar<trtmc::openpi::OpenPIPostAttentionRmsNormCreator>
    plugin_registrar_openpi_post_attention_rms_norm{};

extern "C" void trtmc_openpi_rms_norm_plugin_force_link() {}
