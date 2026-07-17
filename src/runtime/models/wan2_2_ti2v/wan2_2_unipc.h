/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Family-owned order-2 BH2 UniPC flow scheduler used by the native Wan2.2
// TI2V-5B pipeline.  This is an independent C++ implementation of the
// algorithm shipped by the upstream Wan2.2 release; it has no Python, PyTorch,
// or Wan2.1 runtime dependency.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc::wan2_2_ti2v {

class FlowUniPC {
  public:
    explicit FlowUniPC(int32_t num_inference_steps, float shift = 5.0F,
                       int32_t num_train_timesteps = 1000)
        : num_steps_(num_inference_steps), train_steps_(num_train_timesteps) {
        if (num_steps_ <= 0)
            throw std::invalid_argument("Wan2.2 UniPC requires at least one inference step");
        if (train_steps_ <= 1)
            throw std::invalid_argument("Wan2.2 UniPC requires at least two training steps");
        if (!(shift > 0.0F))
            throw std::invalid_argument("Wan2.2 UniPC shift must be positive");
        make_schedule(shift);
    }

    const std::vector<int64_t>& timesteps() const { return timesteps_; }
    const std::vector<float>& sigmas() const { return sigmas_; }
    int32_t step_index() const { return step_index_; }

    void reset() {
        step_index_ = 0;
        lower_order_nums_ = 0;
        previous_order_ = 0;
        model_history_.clear();
        last_sample_.clear();
    }

    void step(const float* model_output, const float* sample, float* output, std::size_t count) {
        if (model_output == nullptr || sample == nullptr || output == nullptr)
            throw std::invalid_argument("Wan2.2 UniPC received a null tensor pointer");
        if (count == 0)
            throw std::invalid_argument("Wan2.2 UniPC received an empty tensor");
        if (step_index_ >= num_steps_)
            throw std::out_of_range("Wan2.2 UniPC has no remaining steps");

        const std::size_t index = static_cast<std::size_t>(step_index_);
        std::vector<float> converted(count);
        const float sigma = sigmas_[index];
        for (std::size_t i = 0; i < count; ++i)
            converted[i] = sample[i] - sigma * model_output[i];

        std::vector<float> corrected(sample, sample + count);
        if (step_index_ > 0 && previous_order_ > 0 && !last_sample_.empty())
            correct(converted, corrected);

        if (model_history_.size() == 2)
            model_history_.erase(model_history_.begin());
        model_history_.push_back(std::move(converted));

        const int32_t remaining = num_steps_ - step_index_;
        const int32_t available_order = std::min<int32_t>(2, lower_order_nums_ + 1);
        const int32_t order = std::min(remaining, available_order);

        last_sample_ = corrected;
        predict(corrected, output, count, order);

        previous_order_ = order;
        lower_order_nums_ = std::min<int32_t>(2, lower_order_nums_ + 1);
        ++step_index_;
    }

  private:
    static float lambda(float sigma) {
        if (sigma == 0.0F)
            return std::numeric_limits<float>::infinity();
        return std::log(1.0F - sigma) - std::log(sigma);
    }

    void make_schedule(float shift) {
        // Upstream constructs sigma_max in float32, interpolates in NumPy
        // float64, applies the rational shift, then stores sigmas as float32
        // while truncating the unrounded timesteps to int64.
        const float sigma_max = 1.0F - (1.0F / static_cast<float>(train_steps_));
        timesteps_.reserve(static_cast<std::size_t>(num_steps_));
        sigmas_.reserve(static_cast<std::size_t>(num_steps_) + 1U);
        for (int32_t i = 0; i < num_steps_; ++i) {
            const double fraction = static_cast<double>(i) / static_cast<double>(num_steps_);
            const double base = static_cast<double>(sigma_max) * (1.0 - fraction);
            const double shifted = static_cast<double>(shift) * base /
                                   (1.0 + (static_cast<double>(shift) - 1.0) * base);
            timesteps_.push_back(static_cast<int64_t>(shifted * train_steps_));
            sigmas_.push_back(static_cast<float>(shifted));
        }
        sigmas_.push_back(0.0F);
    }

    void correct(const std::vector<float>& model_t, std::vector<float>& sample) const {
        const int32_t index = step_index_;
        const float sigma_t = sigmas_[static_cast<std::size_t>(index)];
        const float sigma_s0 = sigmas_[static_cast<std::size_t>(index - 1)];
        const float alpha_t = 1.0F - sigma_t;
        const float h = lambda(sigma_t) - lambda(sigma_s0);
        const float hh = -h;
        const float b_h = std::expm1(hh);
        const float h_phi_1 = b_h;
        const auto& m0 = model_history_.back();

        float rho_previous = 0.0F;
        float rho_current = 0.5F;
        float rk = 1.0F;
        const std::vector<float>* older = nullptr;
        if (previous_order_ == 2) {
            if (model_history_.size() < 2)
                throw std::logic_error("Wan2.2 UniPC correction history is incomplete");
            older = &model_history_.front();
            const float lambda_si = lambda(sigmas_[static_cast<std::size_t>(index - 2)]);
            rk = (lambda_si - lambda(sigma_s0)) / h;

            float phi_k = h_phi_1 / hh - 1.0F;
            const float b0 = phi_k / b_h;
            phi_k = phi_k / hh - 0.5F;
            const float b1 = phi_k * 2.0F / b_h;
            rho_previous = (b0 - b1) / (1.0F - rk);
            rho_current = b0 - rho_previous;
        }

        for (std::size_t i = 0; i < sample.size(); ++i) {
            float correction = rho_current * (model_t[i] - m0[i]);
            if (older != nullptr)
                correction += rho_previous * (((*older)[i] - m0[i]) / rk);
            const float base = (sigma_t / sigma_s0) * last_sample_[i] - alpha_t * h_phi_1 * m0[i];
            sample[i] = base - alpha_t * b_h * correction;
        }
    }

    void predict(const std::vector<float>& sample, float* output, std::size_t count,
                 int32_t order) const {
        const std::size_t index = static_cast<std::size_t>(step_index_);
        const float sigma_t = sigmas_[index + 1U];
        const float sigma_s0 = sigmas_[index];
        const float alpha_t = 1.0F - sigma_t;
        const float h = lambda(sigma_t) - lambda(sigma_s0);
        const float h_phi_1 = std::expm1(-h);
        const float b_h = h_phi_1;
        const auto& m0 = model_history_.back();

        float rk = 1.0F;
        const std::vector<float>* previous = nullptr;
        if (order == 2) {
            if (model_history_.size() < 2)
                throw std::logic_error("Wan2.2 UniPC prediction history is incomplete");
            previous = &model_history_.front();
            const float lambda_si = lambda(sigmas_[index - 1U]);
            rk = (lambda_si - lambda(sigma_s0)) / h;
        }

        for (std::size_t i = 0; i < count; ++i) {
            float predictor_residual = 0.0F;
            if (previous != nullptr)
                predictor_residual = 0.5F * (((*previous)[i] - m0[i]) / rk);
            const float base = (sigma_t / sigma_s0) * sample[i] - alpha_t * h_phi_1 * m0[i];
            output[i] = base - alpha_t * b_h * predictor_residual;
        }
    }

    int32_t num_steps_{0};
    int32_t train_steps_{1000};
    int32_t step_index_{0};
    int32_t lower_order_nums_{0};
    int32_t previous_order_{0};
    std::vector<int64_t> timesteps_;
    std::vector<float> sigmas_;
    std::vector<std::vector<float>> model_history_;
    std::vector<float> last_sample_;
};

} // namespace trtmc::wan2_2_ti2v
