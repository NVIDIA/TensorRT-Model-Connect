/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc::stable_diffusion {

// DDIM with eta = 0, which makes sampling deterministic for a given seed.
//
// The builder precomputes alphas_cumprod and ships it in runtime.json, so the
// runtime holds no opinion about the beta schedule: it only walks the product.
class DdimScheduler {
  public:
    DdimScheduler(std::vector<float> alphas_cumprod, int32_t num_train_timesteps,
                  int32_t steps_offset)
        : alphas_cumprod_(std::move(alphas_cumprod)), num_train_timesteps_(num_train_timesteps),
          steps_offset_(steps_offset) {
        if (alphas_cumprod_.size() != static_cast<std::size_t>(num_train_timesteps_))
            throw std::runtime_error("stable_diffusion alphas_cumprod does not match the schedule");
    }

    // Descending timesteps, evenly spaced, matching diffusers' DDIM.
    std::vector<int32_t> timesteps(int32_t steps) const {
        if (steps <= 0 || steps > num_train_timesteps_)
            throw std::invalid_argument("stable_diffusion step count is out of range");
        const int32_t stride = num_train_timesteps_ / steps;
        std::vector<int32_t> out;
        out.reserve(static_cast<std::size_t>(steps));
        for (int32_t i = steps - 1; i >= 0; --i)
            out.push_back(i * stride + steps_offset_);
        return out;
    }

    // One DDIM update. 'previous' is the next timestep in the walk, or -1 at the end.
    void step(const float* noise, float* latents, std::size_t count, int32_t timestep,
              int32_t previous) const {
        const double alpha_t = alpha_at(timestep);
        const double alpha_prev = previous >= 0 ? alpha_at(previous) : 1.0;
        const double sqrt_alpha_t = std::sqrt(alpha_t);
        const double sqrt_one_minus = std::sqrt(1.0 - alpha_t);
        const double sqrt_alpha_prev = std::sqrt(alpha_prev);
        const double direction = std::sqrt(1.0 - alpha_prev);
        for (std::size_t i = 0; i < count; ++i) {
            const double sample = latents[i];
            const double eps = noise[i];
            // predict x0, then re-noise onto the previous timestep
            const double original = (sample - sqrt_one_minus * eps) / sqrt_alpha_t;
            latents[i] = static_cast<float>(sqrt_alpha_prev * original + direction * eps);
        }
    }

  private:
    double alpha_at(int32_t timestep) const {
        if (timestep < 0 || timestep >= num_train_timesteps_)
            throw std::out_of_range("stable_diffusion timestep is outside the schedule");
        return static_cast<double>(alphas_cumprod_[static_cast<std::size_t>(timestep)]);
    }

    std::vector<float> alphas_cumprod_;
    int32_t num_train_timesteps_{1000};
    int32_t steps_offset_{1};
};

} // namespace trtmc::stable_diffusion
