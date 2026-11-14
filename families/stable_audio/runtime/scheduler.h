/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// diffusers' CosineDPMSolverMultistepScheduler, the sampler of Stable Audio Open:
// SDE DPM-Solver++ (first or second order) over an exponential or Karras sigma
// schedule with EDM preconditioning (c_in, c_skip, c_out, c_noise = atan(sigma) * 2 / pi).
// The noise of every step is supplied by the caller.

#include <cstddef>
#include <string>
#include <vector>

namespace trtmc::stable_audio {

struct SchedulerConfig {
    double sigma_min{0.3};
    double sigma_max{500.0};
    double sigma_data{1.0};
    double rho{7.0};
    std::string sigma_schedule{"exponential"};
    std::string prediction_type{"v_prediction"};
    std::string solver_type{"midpoint"};
    std::string final_sigmas_type{"zero"};
    int solver_order{2};
};

class CosineDpmSolver {
  public:
    explicit CosineDpmSolver(SchedulerConfig config);

    // Prepare a fresh run of `steps` denoising steps; resets the solver history.
    void set_timesteps(int steps);

    // steps + 1 noise levels (float32, as diffusers keeps them); the last is the final sigma.
    const std::vector<float>& sigmas() const noexcept { return sigmas_; }
    // The denoiser's timestep input of each step: atan(sigma) * 2 / pi.
    const std::vector<float>& timesteps() const noexcept { return timesteps_; }
    // sqrt(sigma_max^2 + 1), the scale of the initial unit-variance noise.
    float init_noise_sigma() const noexcept;
    // c_in of step `index`: the factor the latents are scaled by before the denoiser.
    float input_scale(int index) const;

    // One step: `model_output` is the (guided) denoiser output at `sample`; `sample`
    // is replaced by the next sample. `noise` holds `count` unit Gaussian values.
    void step(int index, const float* model_output, float* sample, const float* noise,
              std::size_t count);

  private:
    SchedulerConfig config_;
    std::vector<float> sigmas_;
    std::vector<float> timesteps_;
    // The data predictions of the two latest steps (oldest first).
    std::vector<std::vector<float>> history_;
    int lower_order_nums_{0};
};

} // namespace trtmc::stable_audio
