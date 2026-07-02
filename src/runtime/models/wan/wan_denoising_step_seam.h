/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {
namespace wan_denoising {

template <typename ComputeTembFn, typename PrepareHiddenFn, typename PredictNoiseFn,
          typename UnpatchifyNoiseFn, typename ApplySchedulerFn, typename LogStepFn>
bool run_wan_video_denoising_steps(int32_t num_inference_steps,
                                   const std::vector<float>& step_timesteps,
                                   std::vector<float>& latents, std::string& error,
                                   ComputeTembFn&& compute_temb, PrepareHiddenFn&& prepare_hidden,
                                   PredictNoiseFn&& predict_noise,
                                   UnpatchifyNoiseFn&& unpatchify_noise,
                                   ApplySchedulerFn&& apply_scheduler, LogStepFn&& log_step) {
    std::vector<float> temb_6d;
    std::vector<float> time_embed;
    std::vector<float> hidden;
    std::vector<float> denoiser_output;
    std::vector<float> noise_pred_spatial;

    for (int32_t step = 0; step < num_inference_steps; ++step) {
        const float timestep = step_timesteps[static_cast<std::size_t>(step)];
        compute_temb(timestep, temb_6d, time_embed);
        prepare_hidden(latents, hidden);
        if (!predict_noise(hidden, temb_6d, time_embed, denoiser_output, error)) {
            return false;
        }
        unpatchify_noise(denoiser_output, noise_pred_spatial);
        apply_scheduler(noise_pred_spatial, latents, step);
        log_step(step, timestep, latents);
    }
    return true;
}

} // namespace wan_denoising
} // namespace trtmc
