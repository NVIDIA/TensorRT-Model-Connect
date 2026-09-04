/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/canary/runtime/resampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace trtmc {
namespace {

std::int32_t output_length(std::int32_t num_samples, std::int32_t source_rate,
                           std::int32_t target_rate) {
    return static_cast<std::int32_t>(static_cast<std::int64_t>(num_samples) * target_rate /
                                     source_rate);
}

double scaled_sinc(double distance, double cutoff) {
    constexpr double pi = 3.14159265358979323846;
    if (std::abs(distance) < 1.0e-12)
        return cutoff;
    return cutoff * std::sin(pi * distance * cutoff) / (pi * distance * cutoff);
}

double hann_window(double distance, std::int32_t half_taps) {
    constexpr double pi = 3.14159265358979323846;
    const double position =
        (distance + static_cast<double>(half_taps)) / (2.0 * static_cast<double>(half_taps));
    return 0.5 * (1.0 - std::cos(2.0 * pi * position));
}

float resample_at(const float* samples, std::int32_t num_samples, double source_position,
                  double cutoff, std::int32_t half_taps) {
    const auto center = static_cast<std::int32_t>(std::floor(source_position));
    const auto low = std::max(0, center - half_taps + 1);
    const auto high = std::min(num_samples - 1, center + half_taps);
    double value = 0.0;
    double weight_sum = 0.0;
    for (std::int32_t index = low; index <= high; ++index) {
        const auto distance = static_cast<double>(index) - source_position;
        const auto weight = scaled_sinc(distance, cutoff) * hann_window(distance, half_taps);
        value += static_cast<double>(samples[index]) * weight;
        weight_sum += weight;
    }
    return weight_sum > 1.0e-12 ? static_cast<float>(value / weight_sum) : 0.0F;
}

struct PhaseTable {
    std::int32_t rate_gcd{1};
    std::int32_t phase_count{1};
    std::int32_t half_taps{16};
    std::vector<double> weights;
};

PhaseTable make_phase_table(std::int32_t source_rate, std::int32_t target_rate, double cutoff,
                            std::int32_t half_taps) {
    PhaseTable table;
    table.rate_gcd = std::gcd(source_rate, target_rate);
    table.phase_count = target_rate / table.rate_gcd;
    table.half_taps = half_taps;
    const auto tap_count = 2 * half_taps;
    table.weights.resize(static_cast<std::size_t>(table.phase_count) * tap_count);
    for (std::int32_t phase = 0; phase < table.phase_count; ++phase) {
        const auto fraction =
            static_cast<double>(phase * table.rate_gcd) / static_cast<double>(target_rate);
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const auto offset = tap - half_taps + 1;
            const auto distance = static_cast<double>(offset) - fraction;
            table.weights[static_cast<std::size_t>(phase) * tap_count + tap] =
                scaled_sinc(distance, cutoff) * hann_window(distance, half_taps);
        }
    }
    return table;
}

std::vector<float> resample_phases(const float* samples, std::int32_t num_samples,
                                   std::int32_t source_rate, std::int32_t target_rate,
                                   std::int32_t output_start, std::int32_t output_count,
                                   double cutoff, std::int32_t half_taps) {
    const auto table = make_phase_table(source_rate, target_rate, cutoff, half_taps);
    const auto tap_count = 2 * half_taps;
    std::vector<float> output(static_cast<std::size_t>(output_count));
    for (std::int32_t local = 0; local < output_count; ++local) {
        const auto index = output_start + local;
        const auto numerator = static_cast<std::int64_t>(index) * source_rate;
        const auto center = static_cast<std::int32_t>(numerator / target_rate);
        const auto phase = static_cast<std::int32_t>(numerator % target_rate) / table.rate_gcd;
        const auto* weights = table.weights.data() + static_cast<std::size_t>(phase) * tap_count;
        const auto first = std::max(-half_taps + 1, -center);
        const auto last = std::min(half_taps, num_samples - 1 - center);
        double value = 0.0;
        double weight_sum = 0.0;
        for (std::int32_t offset = first; offset <= last; ++offset) {
            const auto weight = weights[offset + half_taps - 1];
            value += static_cast<double>(samples[center + offset]) * weight;
            weight_sum += weight;
        }
        output[static_cast<std::size_t>(local)] =
            weight_sum > 1.0e-12 ? static_cast<float>(value / weight_sum) : 0.0F;
    }
    return output;
}

} // namespace

std::vector<float> resample_linear(const float* samples, std::int32_t num_samples,
                                   std::int32_t source_rate, std::int32_t target_rate) {
    if (num_samples <= 0)
        return {};
    return resample_linear_range(samples, num_samples, source_rate, target_rate, 0,
                                 output_length(num_samples, source_rate, target_rate));
}

std::vector<float> resample_linear_range(const float* samples, std::int32_t num_samples,
                                         std::int32_t source_rate, std::int32_t target_rate,
                                         std::int32_t output_start, std::int32_t output_count) {
    if (samples == nullptr || num_samples <= 0 || source_rate <= 0 || target_rate <= 0)
        throw std::invalid_argument("resampler requires samples and positive dimensions");
    const auto full_length = output_length(num_samples, source_rate, target_rate);
    const auto start = std::clamp(output_start, 0, full_length);
    const auto count = std::clamp(output_count, 0, full_length - start);
    if (count == 0)
        return {};
    if (source_rate == target_rate)
        return {samples + start, samples + start + count};
    constexpr std::int32_t half_taps = 16;
    const auto cutoff =
        std::min(1.0, static_cast<double>(target_rate) / static_cast<double>(source_rate));
    const auto phase_count = target_rate / std::gcd(source_rate, target_rate);
    if (phase_count <= 2048)
        return resample_phases(samples, num_samples, source_rate, target_rate, start, count, cutoff,
                               half_taps);
    std::vector<float> output(static_cast<std::size_t>(count));
    for (std::int32_t local = 0; local < count; ++local) {
        const auto position = static_cast<double>(start + local) * source_rate / target_rate;
        output[static_cast<std::size_t>(local)] =
            resample_at(samples, num_samples, position, cutoff, half_taps);
    }
    return output;
}

} // namespace trtmc
