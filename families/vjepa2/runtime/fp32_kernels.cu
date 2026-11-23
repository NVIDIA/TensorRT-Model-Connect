/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vjepa2/runtime/fp32_kernels.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
namespace trtmc::vjepa2 {
namespace {
struct Moments {
    float mean, variance, count;
};
__device__ Moments accumulate(Moments old, float x) {
    float count = old.count + 1.0F;
    float difference = x - old.mean;
    float mean = old.mean + difference * (1.0F / count);
    return {mean, old.variance + difference * (x - mean), count};
}
__device__ Moments combine(Moments left, Moments right) {
    if (left.count == 0)
        return right;
    if (right.count == 0)
        return left;
    float count = right.count + left.count;
    float reciprocal = 1.0F / count;
    float right_ratio = right.count * reciprocal;
    float left_ratio = left.count * reciprocal;
    float difference = left.mean - right.mean;
    // Round the left product before the right-term FMA. Contracting the
    // other term changes the encoder's FP32 reduction result.
    const float weighted_mean =
        __fmaf_rn(right_ratio, right.mean, __fmul_rn(left_ratio, left.mean));
    return {weighted_mean,
            right.variance + left.variance + difference * difference * right.count * left_ratio,
            count};
}
__global__ void norm_kernel(const float* x, const float* gamma, const float* bias, float* out,
                            int cols, float eps) {
    __shared__ float means[4], variances[4], counts[4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const auto base = static_cast<std::int64_t>(blockIdx.x) * cols;
    Moments m{0, 0, 0};
    for (int group = threadIdx.x; group < (cols + 3) / 4; group += 128) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int column = group * 4 + item;
            if (column < cols)
                m = accumulate(m, x[base + column]);
        }
    }
    for (int distance = 16; distance > 0; distance /= 2) {
        Moments neighbor{__shfl_down_sync(0xffffffff, m.mean, distance),
                         __shfl_down_sync(0xffffffff, m.variance, distance),
                         __shfl_down_sync(0xffffffff, m.count, distance)};
        m = combine(m, neighbor);
    }
    if (lane == 0) {
        means[warp] = m.mean;
        variances[warp] = m.variance;
        counts[warp] = m.count;
    }
    __syncthreads();
    for (int distance = 2; distance > 0; distance /= 2) {
        if (lane == 0 && warp < distance) {
            m = combine(
                m, {means[warp + distance], variances[warp + distance], counts[warp + distance]});
        }
        __syncthreads();
        if (lane == 0 && warp < distance) {
            means[warp] = m.mean;
            variances[warp] = m.variance;
            counts[warp] = m.count;
        }
        __syncthreads();
    }
    float reciprocal_std = rsqrtf(variances[0] / float(cols) + eps);
    for (int col = threadIdx.x; col < cols; col += 128) {
        out[base + col] = gamma[col] * (reciprocal_std * (x[base + col] - means[0])) + bias[col];
    }
}
template <int Extent>
__global__ void softmax_kernel(const float* x, float* out, int rows, int cols) {
    constexpr int Elements = Extent / 32;
    const auto row = static_cast<std::int64_t>(blockIdx.x) * 4 + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows)
        return;
    float values[Elements];
#pragma unroll
    for (int i = 0; i < Elements; ++i)
        values[i] = lane + i * 32 < cols ? x[static_cast<std::int64_t>(row) * cols + lane + i * 32]
                                         : -INFINITY;
    float maximum = values[0];
#pragma unroll
    for (int i = 0; i < Elements; ++i)
        maximum = maximum > values[i] ? maximum : values[i];
    for (int distance = 16; distance > 0; distance /= 2) {
        float other = __shfl_xor_sync(0xffffffff, maximum, distance);
        maximum = maximum < other ? other : maximum;
    }
    float total = 0;
#pragma unroll
    for (int i = 0; i < Elements; ++i) {
        values[i] = expf(values[i] - maximum);
        total += values[i];
    }
    for (int distance = 16; distance > 0; distance /= 2)
        total += __shfl_xor_sync(0xffffffff, total, distance);
#pragma unroll
    for (int i = 0; i < Elements; ++i)
        if (lane + i * 32 < cols)
            out[static_cast<std::int64_t>(row) * cols + lane + i * 32] = values[i] / total;
}
// Keep the FP32 approximation fixed: CUDA libdevice revisions change erf
// rounding enough to affect the encoder after twelve residual blocks.
__device__ float stable_erf(float x) {
    float magnitude = fabsf(x);
    bool tail = magnitude >= 0x1.00c1fcp+0F;
    float variable = tail ? magnitude : x * x;
    float polynomial = tail ? 0x1.d69874p-14F : 0x1.63d2d4p-14F;
    polynomial = __fmaf_rn(polynomial, variable, tail ? -0x1.5c00b6p-10F : -0x1.ae9a4p-11F);
    polynomial = __fmaf_rn(polynomial, variable, tail ? 0x1.13233ep-7F : 0x1.55abd4p-8F);
    polynomial = __fmaf_rn(polynomial, variable, tail ? -0x1.49b334p-5F : -0x1.b837cep-6F);
    polynomial = __fmaf_rn(polynomial, variable, tail ? 0x1.46aa32p-3F : 0x1.ce315ep-4F);
    polynomial = __fmaf_rn(polynomial, variable, tail ? 0x1.d369f2p-1F : -0x1.812758p-2F);
    polynomial = __fmaf_rn(polynomial, variable, tail ? 0x1.421428p-1F : 0x1.06eba6p-3F);
    float signed_variable = tail ? -variable : x;
    float result = __fmaf_rn(polynomial, signed_variable, signed_variable);
    if (tail) {
        float exponential;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(result));
        result = copysignf(1.0F - exponential, x);
    }
    return result;
}
__global__ void gelu_kernel(const float* x, float* y, int count) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        float value = x[index];
        y[index] = value * 0.5F * (1.0F + stable_erf(value * 0.70710678118654752440F));
    }
}

__global__ void softmax_wide(const float* input, float* output, int rows, int width) {
    const auto row = static_cast<std::int64_t>(blockIdx.x) * 4 + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows)
        return;
    const auto base = static_cast<std::int64_t>(row) * width;
    float maximum = -INFINITY;
    for (std::int64_t column = lane; column < width; column += 32)
        maximum = fmaxf(maximum, input[base + column]);
    for (int offset = 16; offset > 0; offset /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    float sum = 0;
    for (std::int64_t column = lane; column < width; column += 32)
        sum += expf(input[base + column] - maximum);
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    for (std::int64_t column = lane; column < width; column += 32)
        output[base + column] = __fdiv_rn(expf(input[base + column] - maximum), sum);
}

template <bool Maximum>
__device__ float block_reduce(float value, float* scratch) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other = __shfl_down_sync(0xffffffff, value, offset);
        value = Maximum ? fmaxf(value, other) : value + other;
    }
    // All warps must finish reading the previous broadcast before its
    // storage is reused by the next reduction in this attention row.
    __syncthreads();
    if (lane == 0)
        scratch[warp] = value;
    __syncthreads();
    value = threadIdx.x < 32 ? scratch[lane] : (Maximum ? -INFINITY : 0.0F);
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            const float other = __shfl_down_sync(0xffffffff, value, offset);
            value = Maximum ? fmaxf(value, other) : value + other;
        }
    }
    if (threadIdx.x == 0)
        scratch[0] = value;
    __syncthreads();
    return scratch[0];
}

// One block owns one 8192-token attention row. Keep the reference's
// eight-values-per-thread accumulation and two-stage warp reduction order.
__global__ void softmax_attention(const float* input, float* output) {
    __shared__ float scratch[32];
    const auto base = static_cast<std::int64_t>(blockIdx.x) * 8192 + threadIdx.x;
    float values[8];
    float maximum = -INFINITY;
#pragma unroll
    for (int index = 0; index < 8; ++index) {
        values[index] = input[base + index * 1024];
        maximum = fmaxf(maximum, values[index]);
    }
    maximum = block_reduce<true>(maximum, scratch);
    float total = 0;
#pragma unroll
    for (int index = 0; index < 8; ++index)
        total += expf(values[index] - maximum);
    total = block_reduce<false>(total, scratch);
#pragma unroll
    for (int index = 0; index < 8; ++index)
        output[base + index * 1024] = __fdiv_rn(expf(values[index] - maximum), total);
}

__global__ void rotary_kernel(const float* input, float* output, int count, int length, int width,
                              int grid) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const int column = index % width;
    const int rotary_width = 2 * ((width / 3) / 2);
    if (column >= 3 * rotary_width) {
        output[index] = input[index];
        return;
    }
    const int token = (index / width) % length;
    const int axis = column / rotary_width;
    const int position =
        axis == 0 ? token / (grid * grid) : (axis == 1 ? token / grid % grid : token % grid);
    const int local = column % rotary_width;
    // Tensor/scalar division multiplies by the rounded scalar reciprocal.
    const float exponent =
        float(local % (rotary_width / 2)) * __fdiv_rn(1.0F, float(rotary_width / 2));
    const float omega = __fdiv_rn(1.0F, powf(10000.0F, exponent));
    const float angle = position * omega;
    const float rotated = local % 2 == 0 ? -input[index + 1] : input[index - 1];
    output[index] =
        __fadd_rn(__fmul_rn(input[index], cosf(angle)), __fmul_rn(rotated, sinf(angle)));
}
} // namespace
cudaError_t norm_fp32(const float* input, const float* scale, const float* bias, float* output,
                      int rows, int width, float epsilon, cudaStream_t stream) noexcept {
    if (!input || !scale || !bias || !output || rows <= 0 || width <= 0 || rows > INT_MAX / width ||
        width > 0x1000000 || !std::isfinite(epsilon) || epsilon <= 0)
        return cudaErrorInvalidValue;
    norm_kernel<<<rows, 128, 0, stream>>>(input, scale, bias, output, width, epsilon);
    return cudaGetLastError();
}
cudaError_t softmax_fp32(const float* input, float* output, int rows, int width,
                         cudaStream_t stream) noexcept {
    if (!input || !output || rows <= 0 || width <= 0 || rows > INT_MAX / width)
        return cudaErrorInvalidValue;
    const auto blocks = (static_cast<unsigned>(rows) + 3U) / 4U;
    if (width == 8192)
        softmax_attention<<<rows, 1024, 0, stream>>>(input, output);
    else if (width <= 1024)
        softmax_kernel<1024><<<blocks, dim3(32, 4), 0, stream>>>(input, output, rows, width);
    else if (width <= 2048)
        softmax_kernel<2048><<<blocks, dim3(32, 4), 0, stream>>>(input, output, rows, width);
    else
        softmax_wide<<<blocks, dim3(32, 4), 0, stream>>>(input, output, rows, width);
    return cudaGetLastError();
}
cudaError_t gelu_fp32(const float* input, float* output, int count, cudaStream_t stream) noexcept {
    if (!input || !output || count <= 0)
        return cudaErrorInvalidValue;
    gelu_kernel<<<(static_cast<unsigned>(count) + 255U) / 256U, 256, 0, stream>>>(input, output,
                                                                                  count);
    return cudaGetLastError();
}
cudaError_t rotary_fp32(const float* input, float* output, int heads, int length, int width,
                        int grid, cudaStream_t stream) noexcept {
    if (!input || !output || heads <= 0 || length <= 0 || width < 6 || heads > INT_MAX / length ||
        heads * length > INT_MAX / width || grid <= 0 || grid > 46340 ||
        length % (grid * grid) != 0)
        return cudaErrorInvalidValue;
    const int count = heads * length * width;
    rotary_kernel<<<(static_cast<unsigned>(count) + 255U) / 256U, 256, 0, stream>>>(
        input, output, count, length, width, grid);
    return cudaGetLastError();
}
} // namespace trtmc::vjepa2
