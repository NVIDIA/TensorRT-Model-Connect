/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_audio/runtime/scheduler.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace trtmc::stable_audio {
namespace {

constexpr double kPi = 3.141592653589793;

// torch.linspace(start, end, count) evaluated in double; one point yields `start`.
std::vector<double> linspace(double start, double end, int count) {
    std::vector<double> values(static_cast<std::size_t>(count));
    if (count == 1) {
        values[0] = start;
        return values;
    }
    const double step = (end - start) / static_cast<double>(count - 1);
    for (int index = 0; index < count; ++index)
        values[static_cast<std::size_t>(index)] =
            index < count / 2 ? start + step * index : end - step * (count - 1 - index);
    return values;
}

// lambda = log(alpha) - log(sigma) with alpha = 1 (the inputs are pre-scaled); +inf at sigma 0.
double lambda_of(float sigma) {
    return -std::log(static_cast<double>(sigma));
}

} // namespace

CosineDpmSolver::CosineDpmSolver(SchedulerConfig config) : config_(std::move(config)) {
    if (config_.sigma_schedule != "exponential" && config_.sigma_schedule != "karras")
        throw std::invalid_argument("stable_audio scheduler: unknown sigma_schedule");
    if (config_.prediction_type != "v_prediction" && config_.prediction_type != "epsilon")
        throw std::invalid_argument("stable_audio scheduler: unknown prediction_type");
    if (config_.solver_type != "midpoint" && config_.solver_type != "heun")
        throw std::invalid_argument("stable_audio scheduler: unknown solver_type");
    // "sigma_min" would repeat the schedule's last noise level, where the SDE update is undefined.
    if (config_.final_sigmas_type != "zero")
        throw std::invalid_argument("stable_audio scheduler: final_sigmas_type must be zero");
    if (config_.solver_order != 1 && config_.solver_order != 2)
        throw std::invalid_argument("stable_audio scheduler: solver_order must be 1 or 2");
    if (!(config_.sigma_min > 0.0) || !(config_.sigma_max > config_.sigma_min) ||
        !(config_.sigma_data > 0.0) || !(config_.rho > 0.0))
        throw std::invalid_argument("stable_audio scheduler: invalid noise levels");
}

void CosineDpmSolver::set_timesteps(int steps) {
    if (steps <= 0)
        throw std::invalid_argument("stable_audio scheduler: steps must be positive");
    std::vector<double> levels;
    if (config_.sigma_schedule == "karras") {
        const double min_inv_rho = std::pow(config_.sigma_min, 1.0 / config_.rho);
        const double max_inv_rho = std::pow(config_.sigma_max, 1.0 / config_.rho);
        for (const double ramp : linspace(0.0, 1.0, steps))
            levels.push_back(
                std::pow(max_inv_rho + ramp * (min_inv_rho - max_inv_rho), config_.rho));
    } else {
        // exp(linspace(log sigma_min, log sigma_max)) flipped: from sigma_max down.
        const auto exponents =
            linspace(std::log(config_.sigma_min), std::log(config_.sigma_max), steps);
        for (auto it = exponents.rbegin(); it != exponents.rend(); ++it)
            levels.push_back(std::exp(*it));
    }
    sigmas_.clear();
    timesteps_.clear();
    for (const double level : levels) {
        const auto sigma = static_cast<float>(level);
        sigmas_.push_back(sigma);
        timesteps_.push_back(static_cast<float>(std::atan(static_cast<double>(sigma)) / kPi * 2.0));
    }
    sigmas_.push_back(0.0F);
    history_.clear();
    lower_order_nums_ = 0;
}

float CosineDpmSolver::init_noise_sigma() const noexcept {
    return static_cast<float>(std::sqrt(config_.sigma_max * config_.sigma_max + 1.0));
}

float CosineDpmSolver::input_scale(int index) const {
    const double sigma = sigmas_.at(static_cast<std::size_t>(index));
    return static_cast<float>(1.0 /
                              std::sqrt(sigma * sigma + config_.sigma_data * config_.sigma_data));
}

void CosineDpmSolver::step(int index, const float* model_output, float* sample, const float* noise,
                           std::size_t count) {
    const int steps = static_cast<int>(timesteps_.size());
    if (index < 0 || index >= steps)
        throw std::out_of_range("stable_audio scheduler: step index outside the schedule");
    // diffusers takes a first-order step at the end when euler_at_final, when
    // lower_order_final with fewer than 15 steps, or when the final sigma is zero; the
    // last is always the case here, so the other two flags never change the result.
    // Its "lower_order_second" flag can only select the second-order update below.
    const bool final_step = index == steps - 1;

    // EDM output preconditioning: the data prediction x0 from the model output.
    const double sigma = sigmas_[static_cast<std::size_t>(index)];
    const double data2 = config_.sigma_data * config_.sigma_data;
    const auto skip = static_cast<float>(data2 / (sigma * sigma + data2));
    const double out_scale = sigma * config_.sigma_data / std::sqrt(sigma * sigma + data2);
    const auto out =
        static_cast<float>(config_.prediction_type == "v_prediction" ? -out_scale : out_scale);
    std::vector<float> data(count);
    for (std::size_t i = 0; i < count; ++i)
        data[i] = skip * sample[i] + out * model_output[i];
    if (static_cast<int>(history_.size()) == config_.solver_order)
        history_.erase(history_.begin());
    history_.push_back(std::move(data));
    const auto& current = history_.back();

    const float sigma_t = sigmas_[static_cast<std::size_t>(index) + 1];
    const float sigma_s0 = sigmas_[static_cast<std::size_t>(index)];
    const double h = lambda_of(sigma_t) - lambda_of(sigma_s0);
    const double decay = std::exp(-h);
    const double damp = 1.0 - std::exp(-2.0 * h);
    const auto keep = static_cast<float>(static_cast<double>(sigma_t) / sigma_s0 * decay);
    const auto lead = static_cast<float>(damp);
    const auto spread = static_cast<float>(static_cast<double>(sigma_t) * std::sqrt(damp));

    if (config_.solver_order == 1 || lower_order_nums_ < 1 || final_step) {
        for (std::size_t i = 0; i < count; ++i)
            sample[i] = keep * sample[i] + lead * current[i] + spread * noise[i];
    } else {
        const float sigma_s1 = sigmas_[static_cast<std::size_t>(index) - 1];
        const double h0 = lambda_of(sigma_s0) - lambda_of(sigma_s1);
        const auto inverse_r0 = static_cast<float>(1.0 / (h0 / h));
        const auto& previous = history_.front();
        const auto correction = static_cast<float>(
            config_.solver_type == "midpoint" ? 0.5 * damp : damp / (-2.0 * h) + 1.0);
        for (std::size_t i = 0; i < count; ++i) {
            const float derivative = inverse_r0 * (current[i] - previous[i]);
            sample[i] =
                keep * sample[i] + lead * current[i] + correction * derivative + spread * noise[i];
        }
    }
    if (lower_order_nums_ < config_.solver_order)
        ++lower_order_nums_;
}

} // namespace trtmc::stable_audio
