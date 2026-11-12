/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::vibevoice {

// The multistep DPM-Solver++ the reference samples latents with: diffusers'
// DPMSolverMultistepScheduler with a squaredcos_cap_v2 schedule, v-prediction,
// second-order midpoint updates, a first-order first step and a first-order
// final step to sigma = 0. Scalars are evaluated in float32 like the
// reference's CPU scheduler tensors. With ``bfloat16`` set, every value the
// reference holds in the model's bfloat16 type is rounded where it does.
class DpmSolver {
  public:
    DpmSolver(const std::vector<float>& train_sigmas, std::int32_t steps, bool bfloat16);

    std::int32_t steps() const noexcept { return static_cast<std::int32_t>(timesteps_.size()); }
    const std::vector<std::int64_t>& timesteps() const noexcept { return timesteps_; }

    // Start a new latent; ``sample`` holds the initial noise.
    void reset() noexcept;
    // Advance ``sample`` in place with the guided model output of ``step``.
    void step(std::int32_t step, const std::vector<float>& velocity, std::vector<float>& sample);

  private:
    float round(float value) const noexcept;

    std::vector<std::int64_t> timesteps_;
    std::vector<float> sigmas_;
    bool bfloat16_{false};
    std::vector<float> previous_;
    std::vector<float> current_;
    bool has_previous_{false};
};

// The timesteps of diffusers' "linspace" spacing over ``train_steps``.
std::vector<std::int64_t> linspace_timesteps(std::int32_t train_steps, std::int32_t steps);

} // namespace trtmc::vibevoice
