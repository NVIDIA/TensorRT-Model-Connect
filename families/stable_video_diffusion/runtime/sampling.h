/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::stable_video_diffusion {

// EulerDiscreteScheduler with use_karras_sigmas, continuous v-prediction
// timesteps and a final sigma of zero, evaluated with the reference's float32
// rounding: the Karras ramp in float64, sigmas stored as float32, and each
// scalar coefficient rounded to float32 before it scales a tensor.
struct KarrasSchedule {
    std::vector<float> sigmas;    // steps + 1 values; the last is zero.
    std::vector<float> timesteps; // 0.25 * log(sigma) for the first `steps` sigmas.
};

KarrasSchedule karras_schedule(std::int32_t steps, double sigma_min, double sigma_max);

// sqrt(max_sigma^2 + 1) for "leading" spacing, max_sigma for linspace/trailing.
float init_noise_sigma(const KarrasSchedule& schedule, bool includes_unit);

// The factor scale_model_input multiplies by: 1 / sqrt(sigma^2 + 1).
float input_scale(float sigma);

// One Euler step for a v-prediction: sample <- sample + (sample - x0) / sigma * dt.
void euler_v_step(const float* prediction, float* sample, std::size_t count, float sigma,
                  float sigma_next);

// torch.linspace(minimum, maximum, frames) in float32: one guidance scale per frame.
std::vector<float> guidance_ramp(float minimum, float maximum, std::int32_t frames);

// unconditional + scale[f] * (conditional - unconditional), frame-major
// [frames, frame_size] tensors.
void apply_guidance(const float* unconditional, const float* conditional,
                    const std::vector<float>& scales, std::size_t frame_size, float* guided);

} // namespace trtmc::stable_video_diffusion
