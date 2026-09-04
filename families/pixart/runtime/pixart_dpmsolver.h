/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc::diffusion {

class DPMSolverMultistepState {
  public:
    void set_timesteps(int32_t num_steps, double beta_start = 0.0001, double beta_end = 0.02) {
        if (num_steps <= 0) {
            throw std::invalid_argument("DPMSolver requires at least one inference step");
        }
        constexpr int32_t kTrainTimesteps = 1000;
        alpha_t_.resize(kTrainTimesteps);
        sigma_t_.resize(kTrainTimesteps);
        lambda_t_.resize(kTrainTimesteps);
        double cumulative_alpha = 1.0;
        for (int32_t index = 0; index < kTrainTimesteps; ++index) {
            const double beta = beta_start + static_cast<double>(index) /
                                                 static_cast<double>(kTrainTimesteps - 1) *
                                                 (beta_end - beta_start);
            cumulative_alpha *= 1.0 - beta;
            const auto offset = static_cast<std::size_t>(index);
            alpha_t_[offset] = std::sqrt(cumulative_alpha);
            sigma_t_[offset] = std::sqrt(1.0 - cumulative_alpha);
            lambda_t_[offset] = std::log(alpha_t_[offset] / sigma_t_[offset]);
        }

        timesteps.resize(static_cast<std::size_t>(num_steps));
        for (int32_t index = 0; index < num_steps; ++index) {
            const double fraction = static_cast<double>(index) / static_cast<double>(num_steps);
            timesteps[static_cast<std::size_t>(index)] =
                static_cast<float>(std::round((1.0 - fraction) * (kTrainTimesteps - 1)));
        }
        model_outputs_.clear();
    }

    void step(const float* epsilon, const float* sample, float* output, std::size_t count,
              int32_t step_index) {
        if (step_index < 0 || step_index >= static_cast<int32_t>(timesteps.size())) {
            throw std::out_of_range("DPMSolver step index is outside the timestep schedule");
        }
        const int32_t source_timestep = timestep_at(step_index);
        const auto source = static_cast<std::size_t>(source_timestep);
        record_model_output(epsilon, sample, count, source);

        const bool final_step = step_index + 1 == static_cast<int32_t>(timesteps.size());
        const double target_alpha = final_step ? 1.0 : alpha_at(step_index + 1);
        const double target_sigma = final_step ? 0.0 : sigma_at(step_index + 1);
        const double target_lambda =
            final_step ? std::numeric_limits<double>::infinity() : lambda_at(step_index + 1);
        const double h = target_lambda - lambda_t_[source];
        const double ratio = target_sigma / sigma_t_[source];
        const double first_order_coefficient = -target_alpha * std::expm1(-h);

        const bool use_first_order = step_index == 0 || final_step || model_outputs_.size() < 2;
        if (use_first_order) {
            write_first_order_output(sample, output, count, ratio, first_order_coefficient,
                                     model_outputs_.back());
            return;
        }
        write_second_order_output(sample, output, count, step_index, source, h, ratio,
                                  first_order_coefficient);
    }

    std::vector<float> timesteps;

  private:
    void record_model_output(const float* epsilon, const float* sample, std::size_t count,
                             std::size_t source) {
        std::vector<double> predicted_x0(count);
        for (std::size_t index = 0; index < count; ++index) {
            predicted_x0[index] = (static_cast<double>(sample[index]) -
                                   sigma_t_[source] * static_cast<double>(epsilon[index])) /
                                  alpha_t_[source];
        }
        model_outputs_.push_back(std::move(predicted_x0));
        if (model_outputs_.size() > 2) {
            model_outputs_.erase(model_outputs_.begin());
        }
    }

    static void write_first_order_output(const float* sample, float* output, std::size_t count,
                                         double ratio, double first_order_coefficient,
                                         const std::vector<double>& current_x0) {
        for (std::size_t index = 0; index < count; ++index) {
            output[index] = static_cast<float>(ratio * static_cast<double>(sample[index]) +
                                               first_order_coefficient * current_x0[index]);
        }
    }

    void write_second_order_output(const float* sample, float* output, std::size_t count,
                                   int32_t step_index, std::size_t source, double h, double ratio,
                                   double first_order_coefficient) const {
        const int32_t previous_source_timestep = timestep_at(step_index - 1);
        const double previous_h =
            lambda_t_[source] - lambda_t_[static_cast<std::size_t>(previous_source_timestep)];
        const double r0 = previous_h / h;
        const auto& previous_x0 = model_outputs_[0];
        const auto& current_x0 = model_outputs_[1];
        for (std::size_t index = 0; index < count; ++index) {
            const double derivative = (current_x0[index] - previous_x0[index]) / r0;
            output[index] = static_cast<float>(ratio * static_cast<double>(sample[index]) +
                                               first_order_coefficient * current_x0[index] +
                                               0.5 * first_order_coefficient * derivative);
        }
    }

    int32_t timestep_at(int32_t step_index) const {
        return std::clamp(
            static_cast<int32_t>(std::round(timesteps[static_cast<std::size_t>(step_index)])), 0,
            999);
    }

    double alpha_at(int32_t step_index) const {
        return alpha_t_[static_cast<std::size_t>(timestep_at(step_index))];
    }

    double sigma_at(int32_t step_index) const {
        return sigma_t_[static_cast<std::size_t>(timestep_at(step_index))];
    }

    double lambda_at(int32_t step_index) const {
        return lambda_t_[static_cast<std::size_t>(timestep_at(step_index))];
    }

    std::vector<double> alpha_t_;
    std::vector<double> sigma_t_;
    std::vector<double> lambda_t_;
    std::vector<std::vector<double>> model_outputs_;
};

} // namespace trtmc::diffusion
