/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace trtmc {
namespace diffusion {

template <typename ComputeTembFn, typename PrepareHiddenFn, typename RunDenoiserFn,
          typename UnpackVelocityFn, typename ApplySchedulerFn, typename LogStepFn>
bool run_flux_denoising_steps(int32_t num_inference_steps, const std::vector<float>& step_timesteps,
                              std::vector<float>& latents, std::vector<float>& hidden,
                              std::vector<float>& denoiser_output, std::string& error,
                              ComputeTembFn&& compute_temb, PrepareHiddenFn&& prepare_hidden,
                              RunDenoiserFn&& run_denoiser, UnpackVelocityFn&& unpack_velocity,
                              ApplySchedulerFn&& apply_scheduler, LogStepFn&& log_step) {
    using Clock = std::chrono::steady_clock;
    std::vector<float> temb;
    std::vector<float> velocity;
    double total_temb_ms = 0;
    double total_prep_ms = 0;
    double total_dit_ms = 0;
    double total_unpack_ms = 0;
    double total_sched_ms = 0;
    double total_log_ms = 0;

    for (int32_t step = 0; step < num_inference_steps; ++step) {
        const float timestep = step_timesteps[static_cast<std::size_t>(step)];

        auto t0 = Clock::now();
        compute_temb(timestep, temb);
        auto t1 = Clock::now();
        prepare_hidden(latents, hidden);
        auto t2 = Clock::now();
        if (!run_denoiser(hidden, temb, denoiser_output, error)) {
            return false;
        }
        auto t3 = Clock::now();
        unpack_velocity(denoiser_output, velocity);
        auto t4 = Clock::now();
        apply_scheduler(latents, velocity, step);
        auto t5 = Clock::now();
        log_step(step, latents, velocity, hidden);
        auto t6 = Clock::now();

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        total_temb_ms += ms(t0, t1);
        total_prep_ms += ms(t1, t2);
        total_dit_ms += ms(t2, t3);
        total_unpack_ms += ms(t3, t4);
        total_sched_ms += ms(t4, t5);
        total_log_ms += ms(t5, t6);
    }

    std::cerr << "[flux-perf] --- Denoising breakdown (" << num_inference_steps << " steps) ---\n"
              << "[flux-perf]   temb:      " << total_temb_ms << " ms (avg "
              << total_temb_ms / num_inference_steps << ")\n"
              << "[flux-perf]   prep:      " << total_prep_ms << " ms (avg "
              << total_prep_ms / num_inference_steps << ")\n"
              << "[flux-perf]   DiT fwd:   " << total_dit_ms << " ms (avg "
              << total_dit_ms / num_inference_steps << ")\n"
              << "[flux-perf]   unpack:    " << total_unpack_ms << " ms (avg "
              << total_unpack_ms / num_inference_steps << ")\n"
              << "[flux-perf]   scheduler: " << total_sched_ms << " ms (avg "
              << total_sched_ms / num_inference_steps << ")\n"
              << "[flux-perf]   logging:   " << total_log_ms << " ms (avg "
              << total_log_ms / num_inference_steps << ")\n";
    return true;
}

} // namespace diffusion
} // namespace trtmc
