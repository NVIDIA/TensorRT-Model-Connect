/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vibevoice/runtime/scheduler.h"

#include "families/vibevoice/runtime/noise.h"

#include <cmath>
#include <limits>
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

// Up to this many training steps, numpy's float64 linspace points lie within 2^-31 of the
// exact ones, and an exact point that is not a half-integer lies at least 2^-21 from one.
constexpr std::int32_t kMaxTrainSteps = 1 << 20;

int bit_width(std::uint64_t value) {
    int width = 0;
    for (; value != 0; value >>= 1)
        ++width;
    return width;
}

// floor(log2(numerator / denominator)) for positive integers below 2^41.
int floor_log2_ratio(std::uint64_t numerator, std::uint64_t denominator) {
    const int exponent = bit_width(numerator) - bit_width(denominator);
    const bool reached = exponent >= 0 ? denominator <= (numerator >> exponent)
                                       : denominator <= (numerator << -exponent);
    return reached ? exponent : exponent - 1;
}

// numpy's linspace point index * (stop / steps), both products rounded to float64, then
// rounded half to even, computed in integers: no floating-point rounding mode the host
// selects can change it.
std::int64_t linspace_point(std::uint64_t stop, std::uint64_t steps, std::uint64_t index) {
    const std::uint64_t numerator = index * stop; // the exact point is numerator / steps
    const std::uint64_t whole = numerator / steps;
    const std::uint64_t twice_rest = 2 * (numerator % steps);
    // Not a half-integer: float64 rounding cannot move the point across one.
    if (twice_rest != steps)
        return static_cast<std::int64_t>(whole + (twice_rest > steps ? 1U : 0U));
    // An exact half, whole + 1/2. numpy's point is index * x with x the float64 nearest
    // stop / steps: the 53-bit mantissa nearest stop * 2^shift / steps, ties to even.
    const int exponent = floor_log2_ratio(stop, steps);
    const int shift = 52 - exponent;
    std::uint64_t remainder = stop % (2 * steps);
    for (int bit = 0; bit < shift; ++bit)
        remainder = (2 * remainder) % (2 * steps);
    const bool odd = remainder >= steps; // the parity of the truncated mantissa
    remainder %= steps;
    const bool up = 2 * remainder > steps || (2 * remainder == steps && odd);
    // |x - stop / steps| = error / (steps * 2^shift); x lies above stop / steps when up.
    const std::uint64_t error = up ? steps - remainder : remainder;
    if (error != 0) {
        // index * x leaves the half for its float64 neighbour only when it differs from the
        // half by more than half an ulp of the half, 2^(floor(log2 half) - 53); the half's
        // mantissa is even, so a tie stays on it.
        const int scale = 53 - floor_log2_ratio(numerator, steps) - shift; // at most 1
        const std::uint64_t moved = index * error;
        const bool beyond = scale >= 0 ? (moved << scale) > steps : moved > (steps << -scale);
        if (beyond)
            return static_cast<std::int64_t>(whole + (up ? 1U : 0U));
    }
    return static_cast<std::int64_t>(whole + whole % 2U); // half to even
}

} // namespace

std::vector<std::int64_t> linspace_timesteps(std::int32_t train_steps, std::int32_t steps) {
    if (train_steps > kMaxTrainSteps)
        throw std::invalid_argument("vibevoice schedules hold at most " +
                                    std::to_string(kMaxTrainSteps) + " training steps");
    if (steps <= 0 || steps > train_steps)
        throw std::invalid_argument("vibevoice diffusion_steps must lie in [1, " +
                                    std::to_string(train_steps) + "]");
    // numpy.linspace(0, train_steps - 1, steps + 1).round()[::-1][:-1]; the last point is
    // train_steps - 1 itself, as numpy sets it.
    const auto stop = static_cast<std::uint64_t>(train_steps - 1);
    std::vector<std::int64_t> timesteps;
    timesteps.reserve(static_cast<std::size_t>(steps));
    for (std::int32_t index = steps; index >= 1; --index)
        timesteps.push_back(linspace_point(stop, static_cast<std::uint64_t>(steps),
                                           static_cast<std::uint64_t>(index)));
    return timesteps;
}

void check_schedule(const std::vector<float>& train_sigmas, std::int32_t steps) {
    if (train_sigmas.size() > static_cast<std::size_t>(kMaxTrainSteps))
        throw std::invalid_argument("vibevoice schedules hold at most " +
                                    std::to_string(kMaxTrainSteps) + " training steps");
    const auto timesteps =
        linspace_timesteps(static_cast<std::int32_t>(train_sigmas.size()), steps);
    // step() divides by the log-SNR difference of neighbouring steps (h, and r0 =
    // h_previous / h), so the log-SNR must rise strictly along the descending timesteps.
    float previous = 0.0F;
    for (std::size_t index = 0; index < timesteps.size(); ++index) {
        const float sigma = train_sigmas.at(static_cast<std::size_t>(timesteps[index]));
        // A positive sigma whose float32 square stays finite has a finite log-SNR; another one
        // is refused before its logarithm is taken (an infinite square makes alpha and the
        // scaled sigma 0).
        if (!(sigma > 0.0F) || !std::isfinite(sigma))
            throw std::invalid_argument("vibevoice train_sigmas must be positive and finite");
        if (static_cast<double>(sigma) * static_cast<double>(sigma) >
            static_cast<double>(std::numeric_limits<float>::max()))
            throw std::invalid_argument(
                "vibevoice train_sigmas must not exceed the square root of the float32 maximum");
        const float lambda = alpha_sigma(sigma).lambda;
        if (!std::isfinite(lambda) || (index > 0 && !(lambda > previous)))
            throw std::invalid_argument("vibevoice diffusion_steps=" + std::to_string(steps) +
                                        " repeats a noise level of the " +
                                        std::to_string(train_sigmas.size()) + "-step schedule");
        previous = lambda;
    }
}

DpmSolver::DpmSolver(const std::vector<float>& train_sigmas, std::int32_t steps, bool bfloat16)
    : timesteps_(linspace_timesteps(static_cast<std::int32_t>(train_sigmas.size()), steps)),
      bfloat16_(bfloat16) {
    check_schedule(train_sigmas, steps);
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
    // The midpoint update reads the predictions of earlier steps: a latent of another width
    // needs reset() first, and is refused before any state changes.
    if (!current_.empty() && current_.size() != sample.size())
        throw std::invalid_argument(
            "vibevoice scheduler step changes the latent width; call reset() first");
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
    // The final step goes to sigma 0, where the first-order update is the data prediction
    // itself; it is returned without the logarithm of 0 the general update would take.
    if (sigmas_[index + 1] == 0.0F) {
        sample = current_;
        return;
    }

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
