/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/falcon/triattention_kernels.h"


#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace trtmc {

namespace {

constexpr int kScoreBlockSize = 64;
constexpr int kCompactBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxOffsets = 32;
constexpr float kAbsFloor = 1.0e-8F;

template <typename T>
__device__ inline float load_as_float(const T* ptr, int32_t idx);

template <>
__device__ inline float load_as_float<float>(const float* ptr, int32_t idx) {
    return ptr[idx];
}

template <>
__device__ inline float load_as_float<__half>(const __half* ptr, int32_t idx) {
    return __half2float(ptr[idx]);
}

template <>
__device__ inline float load_as_float<__nv_bfloat16>(const __nv_bfloat16* ptr, int32_t idx) {
    return __bfloat162float(ptr[idx]);
}

template <typename T>
__device__ inline float load_cache_value(const void* base, int32_t idx) {
    const T* typed = static_cast<const T*>(base);
    return load_as_float<T>(typed, idx);
}

__device__ inline float warp_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    return value;
}

template <typename LoadT>
__global__ void triattention_score_candidates_kernel(
    const void* __restrict__ d_cache, int32_t kv_dim, int32_t head_dim,
    bool rope_interleaved, const int32_t* __restrict__ candidate_indices,
    int32_t candidate_count, const int32_t* __restrict__ positions_per_head,
    const float* __restrict__ inv_freq, const float* __restrict__ cos_phase,
    const float* __restrict__ sin_phase, int32_t num_offsets,
    const int32_t* __restrict__ head_offsets, const int32_t* __restrict__ head_cache_indices,
    const float* __restrict__ q_mean_real, const float* __restrict__ q_mean_imag,
    const float* __restrict__ q_abs_mean, const float* __restrict__ freq_scale_sq,
    int32_t kv_head_count, bool disable_mlr, bool disable_trig, bool aggregation_max,
    float* __restrict__ scores_out) {
    const int32_t sampled_idx = blockIdx.x;
    const int32_t candidate_idx = blockIdx.y;
    const int32_t tid = threadIdx.x;
    const int32_t warp_id = tid / kWarpSize;
    const int32_t lane = tid % kWarpSize;
    const int32_t warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;

    const int32_t half_dim = head_dim / 2;
    if (sampled_idx >= kv_head_count || candidate_idx >= candidate_count || num_offsets > kMaxOffsets ||
        half_dim <= 0)
        return;

    __shared__ float shared[(kMaxOffsets + 1) * 4];

    const int32_t row = candidate_indices[candidate_idx];
    const int32_t head_offset = head_offsets[sampled_idx];
    const int32_t row_base = row * kv_dim + head_offset;

    const float* q_real_row = q_mean_real + sampled_idx * half_dim;
    const float* q_imag_row = q_mean_imag + sampled_idx * half_dim;
    const float* q_abs_row = q_abs_mean + sampled_idx * half_dim;
    const float* freq_scale_row = freq_scale_sq + sampled_idx * half_dim;

    float local_additive = 0.0F;
    float local_trig[kMaxOffsets];
    #pragma unroll
    for (int o = 0; o < kMaxOffsets; ++o)
        local_trig[o] = 0.0F;

    for (int32_t d = tid; d < half_dim; d += blockDim.x) {
        float k_rot_real = 0.0F;
        float k_rot_imag = 0.0F;
        if (rope_interleaved) {
            k_rot_real = load_cache_value<LoadT>(d_cache, row_base + 2 * d);
            k_rot_imag = load_cache_value<LoadT>(d_cache, row_base + 2 * d + 1);
        } else {
            k_rot_real = load_cache_value<LoadT>(d_cache, row_base + d);
            k_rot_imag = load_cache_value<LoadT>(d_cache, row_base + half_dim + d);
        }

        const float q_real = q_real_row[d];
        const float q_imag = q_imag_row[d];
        const float q_abs = q_abs_row[d];
        const float freq_scale_sq_val = freq_scale_row[d];
        const float q_mean_abs = sqrtf(fmaxf(q_real * q_real + q_imag * q_imag, kAbsFloor));
        const float k_abs = sqrtf(fmaxf(k_rot_real * k_rot_real + k_rot_imag * k_rot_imag, kAbsFloor));
        const float prod_real = q_real * k_rot_real + q_imag * k_rot_imag;
        const float prod_imag = q_imag * k_rot_real - q_real * k_rot_imag;
        const float extra_coef = disable_mlr ? q_abs : (q_abs - q_mean_abs);
        local_additive += k_abs * extra_coef * freq_scale_sq_val;

        if (!disable_trig) {
            for (int32_t o = 0; o < num_offsets; ++o) {
                const int32_t phase_idx = o * half_dim + d;
                local_trig[o] +=
                    freq_scale_sq_val
                    * (prod_real * cos_phase[phase_idx] - prod_imag * sin_phase[phase_idx]);
            }
        }
    }

    local_additive = warp_sum(local_additive);
    for (int32_t o = 0; o < num_offsets; ++o)
        local_trig[o] = warp_sum(local_trig[o]);

    if (lane == 0) {
        const int32_t base = warp_id * (kMaxOffsets + 1);
        shared[base] = local_additive;
        for (int32_t o = 0; o < num_offsets; ++o)
            shared[base + 1 + o] = local_trig[o];
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_additive = (lane < warp_count) ? shared[lane * (kMaxOffsets + 1)] : 0.0F;
        float block_trig[kMaxOffsets];
        #pragma unroll
        for (int o = 0; o < kMaxOffsets; ++o)
            block_trig[o] = 0.0F;
        for (int32_t o = 0; o < num_offsets; ++o) {
            block_trig[o] =
                (lane < warp_count) ? shared[lane * (kMaxOffsets + 1) + 1 + o] : 0.0F;
        }

        block_additive = warp_sum(block_additive);
        for (int32_t o = 0; o < num_offsets; ++o)
            block_trig[o] = warp_sum(block_trig[o]);

        if (lane == 0) {
            float trig_term = 0.0F;
            if (!disable_trig && num_offsets > 0) {
                if (aggregation_max) {
                    trig_term = block_trig[0];
                    for (int32_t o = 1; o < num_offsets; ++o)
                        trig_term = fmaxf(trig_term, block_trig[o]);
                } else {
                    for (int32_t o = 0; o < num_offsets; ++o)
                        trig_term += block_trig[o];
                    trig_term /= static_cast<float>(num_offsets);
                }
            }
            scores_out[sampled_idx * candidate_count + candidate_idx] = trig_term + block_additive;
        }
    }
}

template <typename T>
__global__ void triattention_compact_rows_kernel(const T* __restrict__ src,
                                                 T* __restrict__ scratch, int32_t kv_dim,
                                                 const int32_t* __restrict__ keep_indices,
                                                 int32_t keep_count, int32_t head_group_width,
                                                 int32_t num_kv_heads) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total =
        static_cast<int64_t>(keep_count) * static_cast<int64_t>(num_kv_heads) * head_group_width;
    if (idx >= total)
        return;

    const int32_t dst_row = static_cast<int32_t>(idx / (num_kv_heads * head_group_width));
    const int32_t rem = static_cast<int32_t>(idx % (num_kv_heads * head_group_width));
    const int32_t kv_head = rem / head_group_width;
    const int32_t elem_in_group = rem % head_group_width;
    const int32_t src_row = keep_indices[kv_head * keep_count + dst_row];
    const int32_t col = kv_head * head_group_width + elem_in_group;
    scratch[static_cast<int64_t>(dst_row) * kv_dim + col] =
        src[static_cast<int64_t>(src_row) * kv_dim + col];
}

template <typename LoadT>
bool launch_score_kernel(const void* d_cache, int32_t kv_dim, int32_t head_dim,
                         bool rope_interleaved,
                         const int32_t* d_candidate_indices, int32_t candidate_count,
                         const int32_t* d_positions_per_head, const float* d_inv_freq,
                         const float* d_cos_phase, const float* d_sin_phase, int32_t num_offsets,
                         const int32_t* d_head_offsets, const int32_t* d_head_cache_indices,
                         const float* d_q_mean_real, const float* d_q_mean_imag,
                         const float* d_q_abs_mean, const float* d_freq_scale_sq,
                         int32_t kv_head_count,
                         bool disable_mlr, bool disable_trig, bool aggregation_max,
                         float* d_scores_out, cudaStream_t stream) {
    dim3 block(kScoreBlockSize);
    dim3 grid(static_cast<unsigned>(kv_head_count), static_cast<unsigned>(candidate_count));
    triattention_score_candidates_kernel<LoadT><<<grid, block, 0, stream>>>(
        d_cache, kv_dim, head_dim, rope_interleaved, d_candidate_indices, candidate_count,
        d_positions_per_head, d_inv_freq, d_cos_phase, d_sin_phase, num_offsets, d_head_offsets,
        d_head_cache_indices, d_q_mean_real, d_q_mean_imag, d_q_abs_mean, d_freq_scale_sq,
        kv_head_count, disable_mlr, disable_trig, aggregation_max, d_scores_out);
    return cudaGetLastError() == cudaSuccess;
}

template <typename T>
bool launch_compact_kernel(const void* d_src, void* d_scratch, int32_t kv_dim,
                           const int32_t* d_keep_indices, int32_t keep_count, int32_t head_group_width,
                           int32_t num_kv_heads,
                           cudaStream_t stream) {
    const int64_t total =
        static_cast<int64_t>(keep_count) * static_cast<int64_t>(num_kv_heads) * head_group_width;
    const int blocks = static_cast<int>((total + kCompactBlockSize - 1) / kCompactBlockSize);
    triattention_compact_rows_kernel<T><<<blocks, kCompactBlockSize, 0, stream>>>(
        static_cast<const T*>(d_src), static_cast<T*>(d_scratch), kv_dim, d_keep_indices, keep_count,
        head_group_width, num_kv_heads);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool falcon_triattention_score_candidates_gpu(
    const void* d_cache, DType cache_dtype, int32_t kv_dim, int32_t head_dim,
    bool rope_interleaved, const int32_t* d_candidate_indices, int32_t candidate_count,
    const int32_t* d_positions_per_head, const float* d_inv_freq, const float* d_cos_phase,
    const float* d_sin_phase, int32_t num_offsets, const int32_t* d_head_offsets,
    const int32_t* d_head_cache_indices, const float* d_q_mean_real,
    const float* d_q_mean_imag,
    const float* d_q_abs_mean, const float* d_freq_scale_sq, int32_t kv_head_count, bool disable_mlr,
    bool disable_trig,
    bool aggregation_max, float* d_scores_out, cudaStream_t stream) {
    if (candidate_count <= 0 || kv_head_count <= 0 || head_dim <= 0)
        return false;
    if (num_offsets <= 0 || num_offsets > kMaxOffsets)
        return false;
    (void)d_positions_per_head;
    (void)d_inv_freq;

    switch (cache_dtype) {
    case DType::kFloat32:
        return launch_score_kernel<float>(
            d_cache, kv_dim, head_dim, rope_interleaved, d_candidate_indices, candidate_count,
            d_positions_per_head, d_inv_freq, d_cos_phase, d_sin_phase, num_offsets, d_head_offsets,
            d_head_cache_indices, d_q_mean_real, d_q_mean_imag,
            d_q_abs_mean, d_freq_scale_sq, kv_head_count, disable_mlr, disable_trig, aggregation_max,
            d_scores_out, stream);
    case DType::kFloat16:
        return launch_score_kernel<__half>(
            d_cache, kv_dim, head_dim, rope_interleaved, d_candidate_indices, candidate_count,
            d_positions_per_head, d_inv_freq, d_cos_phase, d_sin_phase, num_offsets, d_head_offsets,
            d_head_cache_indices, d_q_mean_real, d_q_mean_imag,
            d_q_abs_mean, d_freq_scale_sq, kv_head_count, disable_mlr, disable_trig, aggregation_max,
            d_scores_out, stream);
    case DType::kBFloat16:
        return launch_score_kernel<__nv_bfloat16>(
            d_cache, kv_dim, head_dim, rope_interleaved, d_candidate_indices, candidate_count,
            d_positions_per_head, d_inv_freq, d_cos_phase, d_sin_phase, num_offsets, d_head_offsets,
            d_head_cache_indices, d_q_mean_real, d_q_mean_imag,
            d_q_abs_mean, d_freq_scale_sq, kv_head_count, disable_mlr, disable_trig, aggregation_max,
            d_scores_out, stream);
    default:
        return false;
    }
}

bool falcon_triattention_compact_rows_gpu(const void* d_src, void* d_scratch, DType cache_dtype,
                                   int32_t kv_dim, const int32_t* d_keep_indices, int32_t keep_count,
                                   int32_t head_dim, int32_t num_kv_heads, int32_t query_group_size,
                                   cudaStream_t stream) {
    if (keep_count <= 0 || kv_dim <= 0)
        return false;
    if (head_dim <= 0 || num_kv_heads <= 0 || query_group_size <= 0)
        return false;
    const int32_t head_group_width = head_dim * query_group_size;

    switch (cache_dtype) {
    case DType::kFloat32:
        return launch_compact_kernel<float>(d_src, d_scratch, kv_dim, d_keep_indices, keep_count,
                                            head_group_width, num_kv_heads, stream);
    case DType::kFloat16:
        return launch_compact_kernel<__half>(d_src, d_scratch, kv_dim, d_keep_indices, keep_count,
                                             head_group_width, num_kv_heads, stream);
    case DType::kBFloat16:
        return launch_compact_kernel<__nv_bfloat16>(d_src, d_scratch, kv_dim, d_keep_indices,
                                                    keep_count, head_group_width, num_kv_heads,
                                                    stream);
    default:
        return false;
    }
}

} // namespace trtmc

