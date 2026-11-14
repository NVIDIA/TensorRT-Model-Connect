/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Built with -ffp-contract=off: every product and sum below is rounded on its
// own, as the reference evaluates them in separate float32 operations.

#include "families/stable_video_diffusion/runtime/sampling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc::stable_video_diffusion {

namespace {

constexpr double kRho = 7.0;

} // namespace

KarrasSchedule karras_schedule(std::int32_t steps, double sigma_min, double sigma_max) {
    if (steps < 1)
        throw std::invalid_argument("Stable Video Diffusion needs at least one step");
    if (!(sigma_min > 0.0) || !(sigma_max > sigma_min) || !std::isfinite(sigma_max))
        throw std::invalid_argument("Stable Video Diffusion sigma range is invalid");
    const double min_inv_rho = std::pow(sigma_min, 1.0 / kRho);
    const double max_inv_rho = std::pow(sigma_max, 1.0 / kRho);
    // numpy.linspace(0, 1, steps): i * (1 / (steps - 1)), with the end point exact.
    const double ramp_step = steps > 1 ? 1.0 / static_cast<double>(steps - 1) : 0.0;
    KarrasSchedule schedule;
    schedule.sigmas.reserve(static_cast<std::size_t>(steps) + 1);
    schedule.timesteps.reserve(static_cast<std::size_t>(steps));
    for (std::int32_t index = 0; index < steps; ++index) {
        const double ramp = steps > 1 && index == steps - 1 ? 1.0 : index * ramp_step;
        const double sigma = std::pow(max_inv_rho + ramp * (min_inv_rho - max_inv_rho), kRho);
        schedule.sigmas.push_back(static_cast<float>(sigma));
    }
    for (const float sigma : schedule.sigmas)
        schedule.timesteps.push_back(0.25F * std::log(sigma));
    schedule.sigmas.push_back(0.0F);
    return schedule;
}

float init_noise_sigma(const KarrasSchedule& schedule, bool includes_unit) {
    if (schedule.sigmas.empty())
        throw std::invalid_argument("Stable Video Diffusion schedule is empty");
    const float largest = *std::max_element(schedule.sigmas.begin(), schedule.sigmas.end());
    return includes_unit ? std::sqrt(largest * largest + 1.0F) : largest;
}

float input_scale(float sigma) {
    // A CUDA division by a host scalar multiplies by the scalar's reciprocal.
    return 1.0F / std::sqrt(sigma * sigma + 1.0F);
}

void euler_v_step(const float* prediction, float* sample, std::size_t count, float sigma,
                  float sigma_next) {
    // denoised = prediction * c_out + sample * c_skip, c_out = -sigma / sqrt(sigma^2 + 1),
    // c_skip = 1 / (sigma^2 + 1); derivative = (sample - denoised) / sigma.
    const float c_out = -sigma / std::sqrt(sigma * sigma + 1.0F);
    const float c_skip = 1.0F / (sigma * sigma + 1.0F);
    const float inverse_sigma = 1.0F / sigma;
    const float dt = sigma_next - sigma;
    for (std::size_t index = 0; index < count; ++index) {
        const float denoised = prediction[index] * c_out + sample[index] * c_skip;
        const float derivative = (sample[index] - denoised) * inverse_sigma;
        sample[index] = sample[index] + derivative * dt;
    }
}

std::vector<float> guidance_ramp(float minimum, float maximum, std::int32_t frames) {
    if (frames < 1)
        throw std::invalid_argument("Stable Video Diffusion needs at least one frame");
    std::vector<float> scales(static_cast<std::size_t>(frames));
    if (frames == 1) {
        scales[0] = minimum;
        return scales;
    }
    // ATen's linspace: the first half counts up from the start, the second half
    // counts down from the end.
    const float step = (maximum - minimum) / static_cast<float>(frames - 1);
    const std::int32_t halfway = frames / 2;
    for (std::int32_t index = 0; index < frames; ++index) {
        scales[static_cast<std::size_t>(index)] =
            index < halfway ? minimum + step * static_cast<float>(index)
                            : maximum - step * static_cast<float>(frames - index - 1);
    }
    return scales;
}

void apply_guidance(const float* unconditional, const float* conditional,
                    const std::vector<float>& scales, std::size_t frame_size, float* guided) {
    for (std::size_t frame = 0; frame < scales.size(); ++frame) {
        const float scale = scales[frame];
        const std::size_t offset = frame * frame_size;
        for (std::size_t index = offset; index < offset + frame_size; ++index)
            guided[index] =
                unconditional[index] + scale * (conditional[index] - unconditional[index]);
    }
}

} // namespace trtmc::stable_video_diffusion
