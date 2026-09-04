/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc {

void elf_prepare_model_latent(float* model_latent, const float* z, const float* self_condition,
                              const float* condition_mask, int32_t max_length, int32_t text_dim,
                              int32_t input_dim, bool zero_condition, bool zero_self_condition,
                              cudaStream_t stream);

void elf_prepare_sde_latent(float* z_eval, const float* z, const float* noise,
                            const float* condition, const float* condition_mask, float alpha,
                            int32_t max_length, int32_t text_dim, cudaStream_t stream);

void elf_update_latent(float* z, float* self_condition, const float* z_eval, const float* denoised,
                       const float* condition, const float* condition_mask, float timestep,
                       float next_timestep, float timestep_epsilon, int32_t max_length,
                       int32_t text_dim, cudaStream_t stream);

void elf_update_latent_cfg(float* z, float* self_condition, const float* z_eval,
                           const float* conditional_denoised, const float* unconditional_denoised,
                           const float* condition, const float* condition_mask, float cfg_scale,
                           float timestep, float next_timestep, float timestep_epsilon,
                           int32_t max_length, int32_t text_dim, cudaStream_t stream);

void elf_argmax_rows(const float* logits, int32_t vocab_size, int32_t start_row, int32_t row_count,
                     int32_t* token_ids, cudaStream_t stream);

} // namespace trtmc
