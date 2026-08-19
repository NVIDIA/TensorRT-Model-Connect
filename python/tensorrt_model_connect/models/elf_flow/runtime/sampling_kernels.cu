/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling_kernels.h"

#include <cfloat>
#include <climits>
#include <cuda_runtime.h>

namespace trtmc {

namespace {

constexpr int32_t kBlockSize = 256;

__device__ bool candidate_is_better(float lhs_value, int32_t lhs_index, float rhs_value,
                                    int32_t rhs_index) {
    return lhs_value > rhs_value || (lhs_value == rhs_value && lhs_index < rhs_index);
}

__device__ void find_row_candidate(const float* row, int32_t vocab_size, int32_t thread,
                                   float& best_value, int32_t& best_index) {
    best_value = -FLT_MAX;
    best_index = INT_MAX;
    for (int32_t index = thread; index < vocab_size; index += kBlockSize) {
        const float value = row[index];
        if (candidate_is_better(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
}

__device__ void reduce_row_candidates(float* values, int32_t* indices, int32_t thread) {
    for (int32_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (thread < stride &&
            candidate_is_better(values[thread + stride], indices[thread + stride], values[thread],
                                indices[thread])) {
            values[thread] = values[thread + stride];
            indices[thread] = indices[thread + stride];
        }
        __syncthreads();
    }
}

__global__ void prepare_model_latent_kernel(float* model_latent, const float* z,
                                            const float* self_condition,
                                            const float* condition_mask, int32_t numel,
                                            int32_t text_dim, int32_t input_dim,
                                            bool zero_condition, bool zero_self_condition) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numel)
        return;

    const int32_t row = index / text_dim;
    const int32_t column = index - row * text_dim;
    const bool masked = zero_condition && condition_mask[row] > 0.0F;
    const int32_t output_base = row * input_dim + column;
    model_latent[output_base] = masked ? 0.0F : z[index];
    if (input_dim == 2 * text_dim) {
        const float value = zero_self_condition || masked ? 0.0F : self_condition[index];
        model_latent[output_base + text_dim] = value;
    }
}

__global__ void prepare_sde_latent_kernel(float* z_eval, const float* z, const float* noise,
                                          const float* condition, const float* condition_mask,
                                          float alpha, int32_t numel, int32_t text_dim) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numel)
        return;

    const int32_t row = index / text_dim;
    if (condition_mask[row] > 0.0F) {
        z_eval[index] = condition[index];
        return;
    }
    z_eval[index] = alpha * z[index] + (1.0F - alpha) * noise[index];
}

__device__ float next_latent(float z_eval, float denoised, float timestep, float next_timestep,
                             float timestep_epsilon) {
    const float denominator = fmaxf(1.0F - timestep, timestep_epsilon);
    const float velocity = (denoised - z_eval) / denominator;
    return z_eval + (next_timestep - timestep) * velocity;
}

__global__ void update_latent_kernel(float* z, float* self_condition, const float* z_eval,
                                     const float* denoised, const float* condition,
                                     const float* condition_mask, float timestep,
                                     float next_timestep, float timestep_epsilon, int32_t numel,
                                     int32_t text_dim) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numel)
        return;

    const int32_t row = index / text_dim;
    if (condition_mask[row] > 0.0F) {
        z[index] = condition[index];
        self_condition[index] = condition[index];
        return;
    }
    const float x = denoised[index];
    z[index] = next_latent(z_eval[index], x, timestep, next_timestep, timestep_epsilon);
    self_condition[index] = x;
}

__global__ void update_latent_cfg_kernel(float* z, float* self_condition, const float* z_eval,
                                         const float* conditional_denoised,
                                         const float* unconditional_denoised,
                                         const float* condition, const float* condition_mask,
                                         float cfg_scale, float timestep, float next_timestep,
                                         float timestep_epsilon, int32_t numel, int32_t text_dim) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numel)
        return;

    const int32_t row = index / text_dim;
    if (condition_mask[row] > 0.0F) {
        z[index] = condition[index];
        self_condition[index] = condition[index];
        return;
    }
    const float unconditional = unconditional_denoised[index];
    const float x = unconditional + cfg_scale * (conditional_denoised[index] - unconditional);
    z[index] = next_latent(z_eval[index], x, timestep, next_timestep, timestep_epsilon);
    self_condition[index] = x;
}

__global__ void argmax_rows_kernel(const float* logits, int32_t vocab_size, int32_t start_row,
                                   int32_t* token_ids) {
    __shared__ float values[kBlockSize];
    __shared__ int32_t indices[kBlockSize];

    const int32_t thread = threadIdx.x;
    const float* row = logits + static_cast<int64_t>(start_row + blockIdx.x) * vocab_size;
    float best_value;
    int32_t best_index;
    find_row_candidate(row, vocab_size, thread, best_value, best_index);
    values[thread] = best_value;
    indices[thread] = best_index;
    __syncthreads();
    reduce_row_candidates(values, indices, thread);
    if (thread == 0)
        token_ids[blockIdx.x] = indices[0];
}

int32_t block_count(int32_t numel) {
    return (numel + kBlockSize - 1) / kBlockSize;
}

} // namespace

void elf_prepare_model_latent(float* model_latent, const float* z, const float* self_condition,
                              const float* condition_mask, int32_t max_length, int32_t text_dim,
                              int32_t input_dim, bool zero_condition, bool zero_self_condition,
                              cudaStream_t stream) {
    const int32_t numel = max_length * text_dim;
    if (!model_latent || !z || !self_condition || !condition_mask || numel <= 0)
        return;
    prepare_model_latent_kernel<<<block_count(numel), kBlockSize, 0, stream>>>(
        model_latent, z, self_condition, condition_mask, numel, text_dim, input_dim, zero_condition,
        zero_self_condition);
}

void elf_prepare_sde_latent(float* z_eval, const float* z, const float* noise,
                            const float* condition, const float* condition_mask, float alpha,
                            int32_t max_length, int32_t text_dim, cudaStream_t stream) {
    const int32_t numel = max_length * text_dim;
    if (!z_eval || !z || !noise || !condition || !condition_mask || numel <= 0)
        return;
    prepare_sde_latent_kernel<<<block_count(numel), kBlockSize, 0, stream>>>(
        z_eval, z, noise, condition, condition_mask, alpha, numel, text_dim);
}

void elf_update_latent(float* z, float* self_condition, const float* z_eval, const float* denoised,
                       const float* condition, const float* condition_mask, float timestep,
                       float next_timestep, float timestep_epsilon, int32_t max_length,
                       int32_t text_dim, cudaStream_t stream) {
    const int32_t numel = max_length * text_dim;
    if (!z || !self_condition || !z_eval || !denoised || !condition || !condition_mask ||
        numel <= 0) {
        return;
    }
    update_latent_kernel<<<block_count(numel), kBlockSize, 0, stream>>>(
        z, self_condition, z_eval, denoised, condition, condition_mask, timestep, next_timestep,
        timestep_epsilon, numel, text_dim);
}

void elf_update_latent_cfg(float* z, float* self_condition, const float* z_eval,
                           const float* conditional_denoised, const float* unconditional_denoised,
                           const float* condition, const float* condition_mask, float cfg_scale,
                           float timestep, float next_timestep, float timestep_epsilon,
                           int32_t max_length, int32_t text_dim, cudaStream_t stream) {
    const int32_t numel = max_length * text_dim;
    if (!z || !self_condition || !z_eval || !conditional_denoised || !unconditional_denoised ||
        !condition || !condition_mask || numel <= 0) {
        return;
    }
    update_latent_cfg_kernel<<<block_count(numel), kBlockSize, 0, stream>>>(
        z, self_condition, z_eval, conditional_denoised, unconditional_denoised, condition,
        condition_mask, cfg_scale, timestep, next_timestep, timestep_epsilon, numel, text_dim);
}

void elf_argmax_rows(const float* logits, int32_t vocab_size, int32_t start_row, int32_t row_count,
                     int32_t* token_ids, cudaStream_t stream) {
    if (!logits || !token_ids || vocab_size <= 0 || row_count <= 0)
        return;
    argmax_rows_kernel<<<row_count, kBlockSize, 0, stream>>>(logits, vocab_size, start_row,
                                                             token_ids);
}

} // namespace trtmc
