/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen_image/qwen_image_scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>

namespace trtmc {

namespace {

// Mirrors the common dynamic-shift formula used by flow-match schedulers.
// Linear interpolation: mu = image_seq_len * m + b where
//   m = (max_shift - base_shift) / (max_seq - base_seq)
//   b = base_shift - m * base_seq
inline double calculate_shift(int32_t image_seq_len, int32_t base_seq, int32_t max_seq,
                              double base_shift, double max_shift) {
    if (max_seq == base_seq) {
        return base_shift;
    }
    const double m = (max_shift - base_shift) / static_cast<double>(max_seq - base_seq);
    const double b = base_shift - m * static_cast<double>(base_seq);
    return static_cast<double>(image_seq_len) * m + b;
}

// Mirrors diffusers _time_shift_exponential:
//   return math.exp(mu) / (math.exp(mu) + (1/t - 1)**sigma)
// We hardcode sigma=1.0 (matches diffusers' set_timesteps call site).
inline double time_shift_exponential(double mu, double t) {
    const double e_mu = std::exp(mu);
    const double inv = (1.0 / t) - 1.0;
    // sigma=1.0 so the exponent simplifies to the identity.
    return e_mu / (e_mu + inv);
}

// Mirrors diffusers _time_shift_linear (provided for completeness).
inline double time_shift_linear(double mu, double t) {
    const double inv = (1.0 / t) - 1.0;
    return mu / (mu + inv);
}

// Mirrors diffusers' stretch_shift_to_terminal():
//   one_minus_z = 1 - t
//   scale = one_minus_z[-1] / (1 - shift_terminal)
//   stretched_t = 1 - (one_minus_z / scale)
inline void apply_shift_terminal(std::vector<double>& sig, double shift_terminal) {
    if (sig.empty()) {
        return;
    }
    const double tail = 1.0 - sig.back();
    if (tail == 0.0) {
        return;
    }
    const double scale = tail / (1.0 - shift_terminal);
    if (scale == 0.0) {
        return;
    }
    for (auto& s : sig) {
        s = 1.0 - ((1.0 - s) / scale);
    }
}

} // namespace

FlowMatchEulerScheduler::FlowMatchEulerScheduler(float shift, int32_t num_train_timesteps) {
    cfg_.shift = shift;
    cfg_.num_train_timesteps = num_train_timesteps;
    cfg_.use_dynamic_shifting = false;
}

FlowMatchEulerScheduler::FlowMatchEulerScheduler(const FlowMatchEulerConfig& cfg) : cfg_(cfg) {}

void FlowMatchEulerScheduler::set_timesteps(int32_t num_steps) {
    // Static-shift path. Match diffusers FlowMatchEulerDiscreteScheduler
    // when use_dynamic_shifting=False:
    // 1. timesteps = linspace(1, N, N)[::-1] then take linspace(t_max, t_min, K)
    // 2. sigmas = timesteps / N
    // 3. sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    // 4. Append terminal sigma=0
    const double n = static_cast<double>(cfg_.num_train_timesteps);
    const double s = static_cast<double>(cfg_.shift);

    // sigma_min = shift * (1/N) / (1 + (shift-1)/N)
    const double raw_sigma_min = 1.0 / n;
    const double sigma_min = s * raw_sigma_min / (1.0 + (s - 1.0) * raw_sigma_min);

    const double t_max = n;
    const double t_min = sigma_min * n;

    // Linspace in t-space
    std::vector<double> t_steps(static_cast<std::size_t>(num_steps));
    for (int32_t i = 0; i < num_steps; ++i) {
        double frac =
            (num_steps <= 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(num_steps - 1);
        t_steps[static_cast<std::size_t>(i)] = t_max + frac * (t_min - t_max);
    }

    // Convert to sigmas
    std::vector<double> sig(static_cast<std::size_t>(num_steps));
    for (int32_t i = 0; i < num_steps; ++i) {
        sig[static_cast<std::size_t>(i)] = t_steps[static_cast<std::size_t>(i)] / n;
    }

    // Apply shift
    if (cfg_.shift != 1.0f) {
        for (auto& sigma : sig) {
            sigma = s * sigma / (1.0 + (s - 1.0) * sigma);
        }
    }

    // shift_terminal applies in both paths.
    if (cfg_.shift_terminal != 0.0f) {
        apply_shift_terminal(sig, static_cast<double>(cfg_.shift_terminal));
    }

    // Append terminal sigma=0
    sigmas_.resize(static_cast<std::size_t>(num_steps) + 1);
    for (int32_t i = 0; i < num_steps; ++i) {
        sigmas_[static_cast<std::size_t>(i)] = static_cast<float>(sig[static_cast<std::size_t>(i)]);
    }
    sigmas_[static_cast<std::size_t>(num_steps)] = 0.0f;

    // Timesteps = sigmas[:-1] * num_train_timesteps
    timesteps_.resize(static_cast<std::size_t>(num_steps));
    for (int32_t i = 0; i < num_steps; ++i) {
        timesteps_[static_cast<std::size_t>(i)] =
            sigmas_[static_cast<std::size_t>(i)] * static_cast<float>(cfg_.num_train_timesteps);
    }
}

void FlowMatchEulerScheduler::set_timesteps(int32_t num_steps, int32_t image_seq_len) {
    if (!cfg_.use_dynamic_shifting) {
        // Image-seq-len argument is irrelevant; fall back to static-shift path.
        set_timesteps(num_steps);
        return;
    }

    if (num_steps <= 0) {
        throw std::runtime_error("FlowMatchEulerScheduler::set_timesteps: num_steps must be > 0");
    }

    // Mirror the Python debug runner schedule exactly:
    //   sigmas = np.linspace(1.0, 1.0/N, N)
    //   mu = calculate_shift(image_seq_len, base, max, base_shift, max_shift)
    //   scheduler.set_timesteps(sigmas=sigmas, mu=mu)
    //
    // Inside diffusers' set_timesteps with custom sigmas + use_dynamic_shifting:
    //   sigmas = time_shift(mu, 1.0, sigmas)           # exponential by default
    //   if shift_terminal: sigmas = stretch_shift_to_terminal(sigmas)
    //   timesteps = sigmas * num_train_timesteps
    //   sigmas = cat([sigmas, [0]])
    const std::size_t N = static_cast<std::size_t>(num_steps);
    std::vector<double> sig(N);
    const double last = 1.0 / static_cast<double>(num_steps);
    for (std::size_t i = 0; i < N; ++i) {
        const double frac = (N == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(N - 1);
        sig[i] = 1.0 + frac * (last - 1.0);
    }

    const double mu =
        calculate_shift(image_seq_len, cfg_.base_image_seq_len, cfg_.max_image_seq_len,
                        static_cast<double>(cfg_.base_shift), static_cast<double>(cfg_.max_shift));

    const bool linear = (cfg_.time_shift_type == "linear");
    for (auto& s : sig) {
        s = linear ? time_shift_linear(mu, s) : time_shift_exponential(mu, s);
    }

    if (cfg_.shift_terminal != 0.0f) {
        apply_shift_terminal(sig, static_cast<double>(cfg_.shift_terminal));
    }

    sigmas_.resize(N + 1);
    for (std::size_t i = 0; i < N; ++i) {
        sigmas_[i] = static_cast<float>(sig[i]);
    }
    sigmas_[N] = 0.0f;

    timesteps_.resize(N);
    const double train = static_cast<double>(cfg_.num_train_timesteps);
    for (std::size_t i = 0; i < N; ++i) {
        timesteps_[i] = static_cast<float>(sig[i] * train);
    }
}

void FlowMatchEulerScheduler::step(float* latents, const float* velocity, int32_t num_elements,
                                   int32_t step_index) {
    auto si = static_cast<std::size_t>(step_index);
    if (si + 1 >= sigmas_.size())
        return;

    const float sigma = sigmas_[si];
    const float sigma_next = sigmas_[si + 1];
    const float dt = sigma_next - sigma; // negative (sigma decreasing)

    // Euler step: latents += dt * velocity
    for (int32_t i = 0; i < num_elements; ++i) {
        latents[i] += dt * velocity[i];
    }
}

} // namespace trtmc
