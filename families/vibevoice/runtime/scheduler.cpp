/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vibevoice/runtime/scheduler.h"

#include "families/vibevoice/runtime/noise.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace trtmc::vibevoice {
namespace {

struct AlphaSigma {
    float alpha;
    float sigma;
    float lambda;
};

AlphaSigma alpha_sigma(float sigma) {
    const float alpha = 1.0F / std::sqrt(sigma * sigma + 1.0F);
    const float scaled = sigma * alpha;
    return {alpha, scaled, std::log(alpha) - std::log(scaled)};
}

} // namespace

std::vector<std::int64_t> linspace_timesteps(std::int32_t train_steps, std::int32_t steps) {
    if (steps <= 0 || steps > train_steps)
        throw std::invalid_argument("vibevoice diffusion_steps must lie in [1, " +
                                    std::to_string(train_steps) + "]");
    // numpy.linspace(0, train_steps - 1, steps + 1).round()[::-1][:-1]
    const double stop = static_cast<double>(train_steps - 1);
    const double step = stop / static_cast<double>(steps);
    std::vector<std::int64_t> timesteps;
    timesteps.reserve(static_cast<std::size_t>(steps));
    for (std::int32_t index = steps; index >= 1; --index) {
        const double value = index == steps ? stop : static_cast<double>(index) * step;
        timesteps.push_back(static_cast<std::int64_t>(std::nearbyint(value)));
    }
    return timesteps;
}

DpmSolver::DpmSolver(const std::vector<float>& train_sigmas, std::int32_t steps, bool bfloat16)
    : timesteps_(linspace_timesteps(static_cast<std::int32_t>(train_sigmas.size()), steps)),
      bfloat16_(bfloat16) {
    sigmas_.reserve(timesteps_.size() + 1);
    for (const auto timestep : timesteps_)
        sigmas_.push_back(train_sigmas.at(static_cast<std::size_t>(timestep)));
    sigmas_.push_back(0.0F);
}

void DpmSolver::reset() noexcept {
    previous_.clear();
    current_.clear();
    has_previous_ = false;
}

float DpmSolver::round(float value) const noexcept {
    return bfloat16_ ? round_bf16(value) : value;
}

void DpmSolver::step(std::int32_t step, const std::vector<float>& velocity,
                     std::vector<float>& sample) {
    if (step < 0 || step >= steps() || velocity.size() != sample.size())
        throw std::invalid_argument("vibevoice scheduler step is out of range");
    const auto index = static_cast<std::size_t>(step);
    const AlphaSigma source = alpha_sigma(sigmas_[index]);
    // v-prediction to data prediction, in the model's type.
    std::vector<float> data(sample.size());
    for (std::size_t item = 0; item < sample.size(); ++item)
        data[item] =
            round(round(source.alpha * sample[item]) - round(source.sigma * velocity[item]));
    if (!current_.empty()) {
        previous_ = current_;
        has_previous_ = true;
    }
    current_ = data;

    const AlphaSigma target = alpha_sigma(sigmas_[index + 1]);
    const float h = target.lambda - source.lambda;
    const float ratio = target.sigma / source.sigma;
    const float coefficient = target.alpha * (std::exp(-h) - 1.0F);
    const bool first_order = step == 0 || step == steps() - 1 || !has_previous_;
    if (first_order) {
        for (std::size_t item = 0; item < sample.size(); ++item)
            sample[item] = round(ratio * sample[item] - round(coefficient * current_[item]));
        return;
    }
    const AlphaSigma older = alpha_sigma(sigmas_[index - 1]);
    const float r0 = (source.lambda - older.lambda) / h;
    const float inverse_r0 = 1.0F / r0;
    const float half_coefficient = 0.5F * coefficient;
    for (std::size_t item = 0; item < sample.size(); ++item) {
        const float d1 = round(inverse_r0 * round(current_[item] - previous_[item]));
        const float value = ratio * sample[item] - round(coefficient * current_[item]);
        sample[item] = round(value - round(half_coefficient * d1));
    }
}

} // namespace trtmc::vibevoice
