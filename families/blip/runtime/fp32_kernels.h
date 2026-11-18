/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>

namespace trtmc::blip {
inline constexpr std::size_t kLinearWorkspaceBytes = 32U * 1024U * 1024U;
class LinearContext;
std::shared_ptr<LinearContext> create_linear_context();
int linear_fp32(LinearContext& context, const float* input, const float* weight, const float* bias,
                float* output, int rows, int input_width, int output_width, void* workspace,
                std::size_t workspace_bytes, cudaStream_t stream) noexcept;
cudaError_t norm_fp32(const float* input, const float* scale, const float* bias, float* output,
                      int rows, int width, float epsilon, cudaStream_t stream) noexcept;
cudaError_t softmax_fp32(const float* input, float* output, int rows, int width,
                         cudaStream_t stream) noexcept;
cudaError_t gelu_fp32(const float* input, float* output, int count, cudaStream_t stream) noexcept;
void initialize_fp32_kernels();
} // namespace trtmc::blip

extern "C" const char* trtmc_blip_initialize_fp32_kernels() noexcept;
