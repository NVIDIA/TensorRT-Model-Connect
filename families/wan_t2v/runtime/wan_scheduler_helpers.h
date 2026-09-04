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
#include <utility>
#include <vector>

namespace trtmc {
namespace diffusion {
namespace wan_scheduler {

inline int32_t resolve_requested_steps(int32_t requested, int32_t fallback,
                                       bool zero_uses_fallback) {
    if (requested < 0) {
        return fallback;
    }
    if (requested == 0 && zero_uses_fallback) {
        return fallback;
    }
    return requested;
}

inline float resolve_requested_guidance(float requested, float fallback) {
    return requested < 0.0F ? fallback : requested;
}

struct FlowMatchEulerConfig {
    int32_t num_train_timesteps{1000};
    float shift{1.0F};
    bool use_dynamic_shifting{false};
    float base_shift{0.5F};
    float max_shift{1.15F};
    int32_t base_image_seq_len{256};
    int32_t max_image_seq_len{4096};
    float shift_terminal{0.0F};
    int32_t image_seq_len{4096};
    bool use_empirical_mu{false};
    bool use_zero_sigma_min{false};
};

struct FlowMatchEulerPlan {
    std::vector<double> sigmas;
    std::vector<float> timesteps;
    double dynamic_mu{0.0};
    bool used_dynamic_shifting{false};
};

inline double apply_flow_match_shift(double sigma, double shift) {
    return shift * sigma / (1.0 + (shift - 1.0) * sigma);
}

inline double compute_dynamic_mu(const FlowMatchEulerConfig& config, int32_t num_steps) {
    if (config.use_empirical_mu) {
        // Empirical dynamic-shift formula used by compatible schedulers.
        const double a1 = 8.73809524e-05;
        const double b1 = 1.89833333;
        const double a2 = 0.00016927;
        const double b2 = 0.45666666;
        const double seq = static_cast<double>(config.image_seq_len);
        if (seq > 4300.0) {
            return a2 * seq + b2;
        }
        const double m_200 = a2 * seq + b2;
        const double m_10 = a1 * seq + b1;
        const double a = (m_200 - m_10) / 190.0;
        const double b = m_200 - 200.0 * a;
        return a * static_cast<double>(num_steps) + b;
    }

    // Diffusers linear mu formula parameterized by scheduler config.
    const double base_seq = static_cast<double>(config.base_image_seq_len);
    const double max_seq = static_cast<double>(config.max_image_seq_len);
    const double m =
        (static_cast<double>(config.max_shift) - static_cast<double>(config.base_shift)) /
        (max_seq - base_seq);
    const double b = static_cast<double>(config.base_shift) - m * base_seq;
    return static_cast<double>(config.image_seq_len) * m + b;
}

inline void fill_dynamic_shift_schedule(FlowMatchEulerPlan& plan, int32_t num_steps,
                                        const FlowMatchEulerConfig& config, double N) {
    plan.used_dynamic_shifting = true;
    plan.dynamic_mu = compute_dynamic_mu(config, num_steps);

    const double sigma_max = 1.0;
    const double sigma_min = 1.0 / N;
    const double exp_mu = std::exp(plan.dynamic_mu);

    for (int32_t i = 0; i < num_steps; ++i) {
        const double frac =
            static_cast<double>(i) / static_cast<double>(std::max(num_steps - 1, 1));
        const double raw_sigma = sigma_max + frac * (sigma_min - sigma_max);
        const double raw_sigma_clamped = std::max(raw_sigma, 1e-10);
        const auto idx = static_cast<std::size_t>(i);
        plan.sigmas[idx] = exp_mu / (exp_mu + (1.0 / raw_sigma_clamped - 1.0));
        plan.timesteps[idx] = static_cast<float>(plan.sigmas[idx] * N);
    }
}

inline void fill_shifted_linear_schedule(FlowMatchEulerPlan& plan, int32_t num_steps,
                                         const FlowMatchEulerConfig& config, double N) {
    const double shift = static_cast<double>(config.shift);
    const double raw_sigma_min = config.use_zero_sigma_min ? 0.0 : 1.0 / N;
    const double sigma_min =
        config.use_zero_sigma_min ? 0.0 : apply_flow_match_shift(raw_sigma_min, shift);
    const double t_min = config.use_zero_sigma_min ? 0.0 : sigma_min * N;

    for (int32_t i = 0; i < num_steps; ++i) {
        const double frac =
            static_cast<double>(i) / static_cast<double>(std::max(num_steps - 1, 1));
        double sigma = (N + frac * (t_min - N)) / N;
        if (config.use_zero_sigma_min || std::abs(config.shift - 1.0F) > 1e-6F) {
            sigma = apply_flow_match_shift(sigma, shift);
        }
        const auto idx = static_cast<std::size_t>(i);
        plan.sigmas[idx] = sigma;
        plan.timesteps[idx] = static_cast<float>(sigma * N);
    }
}

inline void apply_terminal_shift(FlowMatchEulerPlan& plan, int32_t num_steps,
                                 const FlowMatchEulerConfig& config, double N) {
    if (config.shift_terminal <= 0.0F || plan.sigmas.empty()) {
        return;
    }
    const double terminal = static_cast<double>(config.shift_terminal);
    const double one_minus_last = 1.0 - plan.sigmas[static_cast<std::size_t>(num_steps - 1)];
    const double denom = 1.0 - terminal;
    if (std::abs(denom) <= 1e-12) {
        return;
    }
    const double scale = one_minus_last / denom;
    if (std::abs(scale) <= 1e-12) {
        return;
    }
    for (int32_t i = 0; i < num_steps; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        auto& sigma = plan.sigmas[idx];
        sigma = 1.0 - ((1.0 - sigma) / scale);
        plan.timesteps[idx] = static_cast<float>(sigma * N);
    }
}

inline FlowMatchEulerPlan build_flow_match_euler_plan(int32_t num_steps,
                                                      const FlowMatchEulerConfig& config) {
    FlowMatchEulerPlan plan;
    plan.sigmas.resize(static_cast<std::size_t>(std::max(num_steps, 0)) + 1, 0.0);
    plan.timesteps.resize(static_cast<std::size_t>(std::max(num_steps, 0)), 0.0F);

    if (num_steps <= 0) {
        return plan;
    }

    const double N = static_cast<double>(config.num_train_timesteps);
    if (config.use_dynamic_shifting) {
        fill_dynamic_shift_schedule(plan, num_steps, config, N);
    } else {
        fill_shifted_linear_schedule(plan, num_steps, config, N);
    }
    apply_terminal_shift(plan, num_steps, config, N);
    return plan;
}

struct FlowMatchEulerState {
    std::vector<double> sigmas;
    std::vector<float> timesteps;
    int32_t num_train_timesteps{1000};
    float shift{1.0F};
    bool use_dynamic_shifting{false};
    float base_shift{0.5F};
    float max_shift{1.15F};
    int32_t base_image_seq_len{256};
    int32_t max_image_seq_len{4096};
    float shift_terminal{0.0F};
    int32_t image_seq_len{4096};
    bool use_empirical_mu{false};
    bool use_zero_sigma_min{false};
    double last_dynamic_mu{0.0};
    bool last_used_dynamic_shifting{false};

    void set_timesteps(int32_t num_steps) {
        FlowMatchEulerConfig config;
        config.num_train_timesteps = num_train_timesteps;
        config.shift = shift;
        config.use_dynamic_shifting = use_dynamic_shifting;
        config.base_shift = base_shift;
        config.max_shift = max_shift;
        config.base_image_seq_len = base_image_seq_len;
        config.max_image_seq_len = max_image_seq_len;
        config.shift_terminal = shift_terminal;
        config.image_seq_len = image_seq_len;
        config.use_empirical_mu = use_empirical_mu;
        config.use_zero_sigma_min = use_zero_sigma_min;

        auto plan = build_flow_match_euler_plan(num_steps, config);
        sigmas = std::move(plan.sigmas);
        timesteps = std::move(plan.timesteps);
        last_dynamic_mu = plan.dynamic_mu;
        last_used_dynamic_shifting = plan.used_dynamic_shifting;
    }

    void step(const float* velocity, const float* sample, float* output, std::size_t count,
              int32_t step_index) const {
        const double sigma = sigmas[static_cast<std::size_t>(step_index)];
        const double sigma_next = sigmas[static_cast<std::size_t>(step_index) + 1];
        const double dt = sigma_next - sigma;
        for (std::size_t i = 0; i < count; ++i) {
            output[i] = static_cast<float>(static_cast<double>(sample[i]) +
                                           dt * static_cast<double>(velocity[i]));
        }
    }
};

// Wan2.1 checkpoints use UniPC order 2 with flow-prediction sigmas.  This is
// the specialized BH2 predictor/corrector needed by that checkpoint, rather
// than a general-purpose UniPC implementation.
struct UniPCFlowState {
    std::vector<float> sigmas;
    std::vector<float> timesteps;
    int32_t num_train_timesteps{1000};
    float flow_shift{3.0F};
    bool lower_order_final{true};

    std::vector<float> older_model_output;
    std::vector<float> previous_model_output;
    std::vector<float> last_sample;
    int32_t lower_order_nums{0};
    int32_t previous_order{0};

    void set_timesteps(int32_t num_steps) {
        sigmas.assign(static_cast<std::size_t>(std::max(num_steps, 0)) + 1, 0.0F);
        timesteps.assign(static_cast<std::size_t>(std::max(num_steps, 0)), 0.0F);
        older_model_output.clear();
        previous_model_output.clear();
        last_sample.clear();
        lower_order_nums = 0;
        previous_order = 0;
        if (num_steps <= 0) {
            return;
        }

        // Diffusers uses linspace(1, 1/N, num_steps + 1)[:-1], applies
        // flow_shift, nudges sigma[0] below one, then truncates timesteps to
        // int64 before passing them to the transformer.
        const double N = static_cast<double>(num_train_timesteps);
        for (int32_t i = 0; i < num_steps; ++i) {
            const double frac = static_cast<double>(i) / static_cast<double>(num_steps);
            const double raw_sigma = 1.0 + frac * (1.0 / N - 1.0);
            double sigma = static_cast<double>(flow_shift) * raw_sigma /
                           (1.0 + (static_cast<double>(flow_shift) - 1.0) * raw_sigma);
            if (i == 0 && std::abs(sigma - 1.0) < 1e-6) {
                sigma -= 1e-6;
            }
            sigmas[static_cast<std::size_t>(i)] = static_cast<float>(sigma);
            timesteps[static_cast<std::size_t>(i)] =
                static_cast<float>(static_cast<int64_t>(sigma * N));
        }
    }

    static float flow_lambda(float sigma) {
        if (sigma <= 0.0F) {
            return std::numeric_limits<float>::infinity();
        }
        return std::log1p(-sigma) - std::log(sigma);
    }

    void apply_corrector(const std::vector<float>& model_output, std::vector<float>& corrected,
                         std::size_t count, int32_t step_index) const {
        const float sigma_t = sigmas[static_cast<std::size_t>(step_index)];
        const float sigma_s0 = sigmas[static_cast<std::size_t>(step_index - 1)];
        const float alpha_t = 1.0F - sigma_t;
        const float h = flow_lambda(sigma_t) - flow_lambda(sigma_s0);
        const float hh = -h;
        const float h_phi_1 = std::expm1(hh);
        const float B_h = h_phi_1;
        const float base_sample_coeff = sigma_t / sigma_s0;
        const float base_model_coeff = -alpha_t * h_phi_1;
        const float correction_coeff = -alpha_t * B_h;

        float older_rho = 0.0F;
        float current_rho = 0.5F;
        float older_rk = 1.0F;
        if (previous_order == 2) {
            const float lambda_si = flow_lambda(sigmas[static_cast<std::size_t>(step_index - 2)]);
            const float lambda_s0 = flow_lambda(sigma_s0);
            older_rk = (lambda_si - lambda_s0) / h;

            float h_phi_k = h_phi_1 / hh - 1.0F;
            const float b0 = h_phi_k / B_h;
            h_phi_k = h_phi_k / hh - 0.5F;
            const float b1 = 2.0F * h_phi_k / B_h;
            older_rho = (b0 - b1) / (1.0F - older_rk);
            current_rho = b0 - older_rho;
        }

        corrected.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            float residual = current_rho * (model_output[i] - previous_model_output[i]);
            if (previous_order == 2) {
                residual +=
                    older_rho * ((older_model_output[i] - previous_model_output[i]) / older_rk);
            }
            corrected[i] = base_sample_coeff * last_sample[i] +
                           base_model_coeff * previous_model_output[i] +
                           correction_coeff * residual;
        }
    }

    void apply_predictor(const float* sample, float* output, std::size_t count, int32_t step_index,
                         int32_t order) const {
        const float sigma_s0 = sigmas[static_cast<std::size_t>(step_index)];
        const float sigma_t = sigmas[static_cast<std::size_t>(step_index + 1)];
        if (sigma_t == 0.0F) {
            std::copy(previous_model_output.begin(), previous_model_output.end(), output);
            return;
        }

        const float alpha_t = 1.0F - sigma_t;
        const float h = flow_lambda(sigma_t) - flow_lambda(sigma_s0);
        const float h_phi_1 = std::expm1(-h);
        const float B_h = h_phi_1;
        const float sample_coeff = sigma_t / sigma_s0;
        const float model_coeff = -alpha_t * h_phi_1;
        const float residual_coeff = -alpha_t * B_h * 0.5F;

        float rk = 1.0F;
        if (order == 2) {
            const float lambda_si = flow_lambda(sigmas[static_cast<std::size_t>(step_index - 1)]);
            rk = (lambda_si - flow_lambda(sigma_s0)) / h;
        }
        for (std::size_t i = 0; i < count; ++i) {
            float value = sample_coeff * sample[i] + model_coeff * previous_model_output[i];
            if (order == 2) {
                value += residual_coeff * ((older_model_output[i] - previous_model_output[i]) / rk);
            }
            output[i] = value;
        }
    }

    void step(const float* velocity, const float* sample, float* output, std::size_t count,
              int32_t step_index) {
        const float sigma = sigmas[static_cast<std::size_t>(step_index)];
        std::vector<float> model_output(count);
        for (std::size_t i = 0; i < count; ++i) {
            model_output[i] = sample[i] - sigma * velocity[i];
        }

        std::vector<float> corrected;
        const float* predictor_sample = sample;
        if (step_index > 0 && !last_sample.empty()) {
            apply_corrector(model_output, corrected, count, step_index);
            predictor_sample = corrected.data();
        }

        older_model_output = std::move(previous_model_output);
        previous_model_output = std::move(model_output);

        int32_t order = std::min(2, lower_order_nums + 1);
        if (lower_order_final) {
            order = std::min(order, static_cast<int32_t>(timesteps.size()) - step_index);
        }
        last_sample.assign(predictor_sample, predictor_sample + count);
        apply_predictor(predictor_sample, output, count, step_index, order);
        lower_order_nums = std::min(lower_order_nums + 1, 2);
        previous_order = order;
    }
};

} // namespace wan_scheduler
} // namespace diffusion
} // namespace trtmc
