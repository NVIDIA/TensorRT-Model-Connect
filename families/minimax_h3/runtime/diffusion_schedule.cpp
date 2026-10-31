/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/pipeline.h"

#include <stdexcept>

namespace trtmc {

MiniMaxH3Schedule make_minimax_h3_schedule(int32_t grid_points, float shift) {
    if (grid_points < 2 || shift <= 0.0F)
        throw std::invalid_argument("MiniMax-H3 schedule arguments are invalid");
    MiniMaxH3Schedule result;
    result.sigmas.reserve(grid_points);
    for (int32_t index = 0; index < grid_points; ++index) {
        const float base = static_cast<float>(1.0 - static_cast<double>(index) / (grid_points - 1));
        const float sigma = shift * base / (1.0F + (shift - 1.0F) * base);
        if (result.sigmas.empty() || sigma != result.sigmas.back())
            result.sigmas.push_back(sigma);
    }
    if (result.sigmas.size() < 2 || result.sigmas.back() != 0.0F)
        throw std::runtime_error("MiniMax-H3 sigma grid collapsed unexpectedly");
    result.timesteps.reserve(result.sigmas.size() - 1);
    for (std::size_t index = 0; index + 1 < result.sigmas.size(); ++index)
        result.timesteps.push_back(1.0F - result.sigmas[index]);
    return result;
}

MiniMaxH3Schedule make_minimax_h3_pdd_schedule(float shift) {
    if (shift != 12.0F && shift != 3.0F)
        throw std::invalid_argument("OmniRef PDD requires the released video/audio shift");
    MiniMaxH3Schedule result;
    for (int32_t node = 0; node <= 32; node += 4) {
        // Match the pinned FP64 linspace and rational shift without contracting
        // the multiplication/subtraction into a fused operation.
        volatile double offset = static_cast<double>(node) * (0.999 / 32.0);
        const double t = node == 32 ? 0.0 : 0.999 - offset;
        volatile double product = t * (static_cast<double>(shift) - 1.0);
        const double denominator = product + 0.999;
        volatile double numerator = static_cast<double>(shift) * t;
        result.sigmas.push_back(static_cast<float>((numerator * 0.999) / denominator));
    }
    for (std::size_t step = 0; step < 8; ++step)
        result.timesteps.push_back(1.0F - result.sigmas[step]);
    return result;
}

void minimax_h3_scheduler_step(float* sample, const float* velocity, std::size_t count,
                               float timestep, float sigma, float sigma_next) {
    if (sample == nullptr || velocity == nullptr || !(sigma > 0.0F))
        throw std::invalid_argument("MiniMax-H3 scheduler received invalid inputs");
    const float sigma_from_timestep = 1.0F - timestep;
    const float ratio = sigma_next / sigma;
    for (std::size_t index = 0; index < count; ++index) {
        const float denoised = sample[index] + sigma_from_timestep * velocity[index];
        sample[index] = ratio * sample[index] + (1.0F - ratio) * denoised;
    }
}

} // namespace trtmc
