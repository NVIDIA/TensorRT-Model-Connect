/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

void require_sana_wm_native_ops();

bool torch_cuda_bfloat16_randn(int32_t channels, int32_t frames, int32_t height, int32_t width,
                               uint64_t seed, float* output, std::string& error);

bool torch_cuda_bfloat16_sana_ucpe_raymats(const float* camera_conditions,
                                           std::size_t camera_condition_count, int32_t frames,
                                           int32_t height, int32_t width, float* raymats,
                                           float* raymats_inv, std::string& error);

bool torch_float32_sana_chunk_plucker(const float* poses, std::size_t pose_count,
                                      const float* intrinsics, std::size_t intrinsics_count,
                                      int32_t num_frames, int32_t chunk_count, int32_t height,
                                      int32_t width, int32_t time_stride, float* output,
                                      std::string& error);

bool torch_cuda_bfloat16_ltx_flow_step(const float* model_output, std::size_t model_output_count,
                                       const float* sample, std::size_t sample_count,
                                       int32_t channels, int32_t frames, int32_t height,
                                       int32_t width, float timestep, float cfg_scale,
                                       const std::vector<float>& sigmas, float* output,
                                       std::string& error);

bool torch_cuda_bfloat16_refiner_mix(const float* clean, const float* noise, std::size_t count,
                                     float sigma, float* output, std::string& error);

bool torch_cuda_bfloat16_refiner_euler_step(const float* sample, const float* denoised,
                                            std::size_t count, float sigma, float sigma_next,
                                            float* output, std::string& error);

} // namespace trtmc
