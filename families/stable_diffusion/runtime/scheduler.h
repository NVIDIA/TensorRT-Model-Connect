/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
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

// Euler-Ancestral over the sigma parameterisation, which is what SDXL ships.
//
// Read off the reference rather than assumed:
//
//   * timesteps are "trailing" spaced: round(T - i * T / steps) - 1, so one
//     step lands on 999 and the walk ends at sigma 0.
//   * the update is ancestral, injecting noise scaled by sigma_up. On the last
//     step sigma_next is 0, so sigma_up is 0 too: a one-step turbo run is
//     deterministic and needs no shared RNG with the reference.
//   * incoming latents are unit-variance and must be lifted to the schedule's
//     starting noise level. DDIM's equivalent factor is 1.0, which is why it
//     has no counterpart there - and why omitting it scored PSNR 7.83.
class EulerAncestralScheduler {
  public:
    EulerAncestralScheduler(std::vector<float> alphas_cumprod, int32_t num_train_timesteps)
        : alphas_cumprod_(std::move(alphas_cumprod)), num_train_timesteps_(num_train_timesteps) {
        if (alphas_cumprod_.size() != static_cast<std::size_t>(num_train_timesteps_))
            throw std::runtime_error("stable_diffusion alphas_cumprod does not match the schedule");
    }

    std::vector<int32_t> timesteps(int32_t steps) const {
        if (steps <= 0 || steps > num_train_timesteps_)
            throw std::invalid_argument("stable_diffusion step count is out of range");
        const double ratio = static_cast<double>(num_train_timesteps_) / steps;
        std::vector<int32_t> out;
        out.reserve(static_cast<std::size_t>(steps));
        for (int32_t i = 0; i < steps; ++i) {
            const double raw = static_cast<double>(num_train_timesteps_) - i * ratio;
            out.push_back(static_cast<int32_t>(std::llround(raw)) - 1);
        }
        return out;
    }

    // sigma = sqrt((1 - alphas_cumprod) / alphas_cumprod)
    double sigma_at(int32_t timestep) const {
        if (timestep < 0 || timestep >= num_train_timesteps_)
            throw std::out_of_range("stable_diffusion timestep is outside the schedule");
        const double alpha =
            static_cast<double>(alphas_cumprod_[static_cast<std::size_t>(timestep)]);
        return std::sqrt((1.0 - alpha) / alpha);
    }

    double init_noise_sigma(const std::vector<std::int32_t>& schedule) const {
        double largest = 0.0;
        for (const auto timestep : schedule)
            largest = std::max(largest, sigma_at(timestep));
        return largest;
    }

    // The model sees the latents divided down by the current noise level.
    void scale_input(const float* latents, float* scaled, std::size_t count,
                     int32_t timestep) const {
        const double sigma = sigma_at(timestep);
        const double factor = 1.0 / std::sqrt(sigma * sigma + 1.0);
        for (std::size_t i = 0; i < count; ++i)
            scaled[i] = static_cast<float>(latents[i] * factor);
    }

    // One ancestral update. 'previous' is the next timestep, or -1 on the last
    // step where sigma_next is 0. 'gaussian' is ignored when sigma_up is 0.
    void step(const float* noise, float* latents, std::size_t count, int32_t timestep,
              int32_t previous, const float* gaussian) const {
        const double sigma = sigma_at(timestep);
        const double sigma_next = previous >= 0 ? sigma_at(previous) : 0.0;
        const double up_squared =
            sigma * sigma > 0.0 ? sigma_next * sigma_next *
                                      (sigma * sigma - sigma_next * sigma_next) / (sigma * sigma)
                                : 0.0;
        const double sigma_up = up_squared > 0.0 ? std::sqrt(up_squared) : 0.0;
        const double down_squared = sigma_next * sigma_next - sigma_up * sigma_up;
        const double sigma_down = down_squared > 0.0 ? std::sqrt(down_squared) : 0.0;
        const double dt = sigma_down - sigma;
        for (std::size_t i = 0; i < count; ++i) {
            const double sample = latents[i];
            // epsilon prediction: recover x0, take the Euler derivative, walk dt
            const double original = sample - sigma * noise[i];
            const double derivative = (sample - original) / sigma;
            double next = sample + derivative * dt;
            if (sigma_up > 0.0 && gaussian != nullptr)
                next += sigma_up * gaussian[i];
            latents[i] = static_cast<float>(next);
        }
    }

  private:
    std::vector<float> alphas_cumprod_;
    int32_t num_train_timesteps_{1000};
};

} // namespace trtmc::stable_diffusion
