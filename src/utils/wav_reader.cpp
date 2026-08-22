/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/wav_reader.h"

#include "trtmc/trtmc_io.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

int32_t compute_output_length(int32_t n_samples, int32_t source_rate, int32_t target_rate) {
    return static_cast<int32_t>(static_cast<int64_t>(n_samples) * target_rate / source_rate);
}

double scaled_sinc(double distance, double cutoff) {
    constexpr double kPi = 3.14159265358979323846;
    if (std::abs(distance) < 1e-12) {
        return cutoff;
    }
    return cutoff * std::sin(kPi * distance * cutoff) / (kPi * distance * cutoff);
}

double hann_window(double distance, int32_t half_taps) {
    constexpr double kPi = 3.14159265358979323846;
    const double win_pos =
        (distance + static_cast<double>(half_taps)) / (2.0 * static_cast<double>(half_taps));
    return 0.5 * (1.0 - std::cos(2.0 * kPi * win_pos));
}

float resample_at_position(const float* samples, int32_t n_samples, double src_pos, double cutoff,
                           int32_t half_taps) {
    const auto center = static_cast<int32_t>(std::floor(src_pos));
    const int32_t lo = std::max(0, center - half_taps + 1);
    const int32_t hi = std::min(n_samples - 1, center + half_taps);

    double acc = 0.0;
    double weight_sum = 0.0;
    for (int32_t j = lo; j <= hi; ++j) {
        const double distance = static_cast<double>(j) - src_pos;
        const double weight = scaled_sinc(distance, cutoff) * hann_window(distance, half_taps);
        acc += static_cast<double>(samples[j]) * weight;
        weight_sum += weight;
    }

    if (weight_sum > 1e-12) {
        return static_cast<float>(acc / weight_sum);
    }
    return 0.0F;
}

struct PolyphaseSincWeights {
    int32_t rate_gcd{1};
    int32_t phase_count{1};
    int32_t half_taps{16};
    std::vector<double> weights;
};

PolyphaseSincWeights build_polyphase_sinc_weights(int32_t source_rate, int32_t target_rate,
                                                  double cutoff, int32_t half_taps) {
    PolyphaseSincWeights table;
    table.rate_gcd = std::gcd(source_rate, target_rate);
    table.phase_count = target_rate / table.rate_gcd;
    table.half_taps = half_taps;
    const int32_t tap_count = 2 * half_taps;
    table.weights.resize(static_cast<std::size_t>(table.phase_count) * tap_count);

    for (int32_t phase = 0; phase < table.phase_count; ++phase) {
        const double fraction =
            static_cast<double>(phase * table.rate_gcd) / static_cast<double>(target_rate);
        for (int32_t tap = 0; tap < tap_count; ++tap) {
            const int32_t offset = tap - half_taps + 1;
            const double distance = static_cast<double>(offset) - fraction;
            table.weights[static_cast<std::size_t>(phase) * tap_count + tap] =
                scaled_sinc(distance, cutoff) * hann_window(distance, half_taps);
        }
    }
    return table;
}

std::vector<float> resample_polyphase_range(const float* samples, int32_t n_samples,
                                            int32_t source_rate, int32_t target_rate,
                                            int32_t output_start, int32_t output_count,
                                            double cutoff, int32_t half_taps) {
    const PolyphaseSincWeights table =
        build_polyphase_sinc_weights(source_rate, target_rate, cutoff, half_taps);
    const int32_t tap_count = 2 * half_taps;
    std::vector<float> resampled(output_count);

    for (int32_t local_index = 0; local_index < output_count; ++local_index) {
        const int32_t i = output_start + local_index;
        const int64_t position_numerator = static_cast<int64_t>(i) * source_rate;
        const int32_t center = static_cast<int32_t>(position_numerator / target_rate);
        const int32_t remainder = static_cast<int32_t>(position_numerator % target_rate);
        const int32_t phase = remainder / table.rate_gcd;
        const double* phase_weights =
            table.weights.data() + static_cast<std::size_t>(phase) * tap_count;

        const int32_t first_offset = std::max(-half_taps + 1, -center);
        const int32_t last_offset = std::min(half_taps, n_samples - 1 - center);
        double acc = 0.0;
        double weight_sum = 0.0;
        for (int32_t offset = first_offset; offset <= last_offset; ++offset) {
            const double weight = phase_weights[offset + half_taps - 1];
            acc += static_cast<double>(samples[center + offset]) * weight;
            weight_sum += weight;
        }
        resampled[static_cast<std::size_t>(local_index)] =
            weight_sum > 1e-12 ? static_cast<float>(acc / weight_sum) : 0.0F;
    }
    return resampled;
}

} // namespace

WavData read_wav(const std::string& path) {
    auto audio = io::read_wav(path);
    return {std::move(audio.samples), audio.sample_rate};
}

std::vector<float> resample_linear(const float* samples, int32_t n_samples, int32_t source_rate,
                                   int32_t target_rate) {
    if (n_samples <= 0)
        return {};
    if (samples == nullptr)
        throw std::invalid_argument("resample input must not be null");
    if (source_rate <= 0 || target_rate <= 0)
        throw std::invalid_argument("resample rates must be positive");
    const int32_t out_len = compute_output_length(n_samples, source_rate, target_rate);
    return resample_linear_range(samples, n_samples, source_rate, target_rate, 0, out_len);
}

std::vector<float> resample_linear_range(const float* samples, int32_t n_samples,
                                         int32_t source_rate, int32_t target_rate,
                                         int32_t output_start, int32_t output_count) {
    if (n_samples <= 0 || output_count <= 0)
        return {};
    if (samples == nullptr)
        throw std::invalid_argument("resample input must not be null");
    if (source_rate <= 0 || target_rate <= 0)
        throw std::invalid_argument("resample rates must be positive");

    const int32_t full_output_length = compute_output_length(n_samples, source_rate, target_rate);
    const int32_t range_start = std::clamp(output_start, 0, full_output_length);
    const int32_t range_count = std::clamp(output_count, 0, full_output_length - range_start);
    if (range_count <= 0)
        return {};
    if (source_rate == target_rate) {
        return std::vector<float>(samples + range_start, samples + range_start + range_count);
    }

    const int32_t half_taps = 16;
    const double cutoff =
        std::min(1.0, static_cast<double>(target_rate) / static_cast<double>(source_rate));

    // Source positions repeat across target_rate / gcd(source_rate,
    // target_rate) fractional phases. Precompute the windowed-sinc weights for
    // those phases instead of evaluating sin/cos for every output sample and
    // tap. Fall back for unusual rate pairs that would require a large table.
    constexpr int32_t kMaxPrecomputedPhases = 2048;
    const int32_t phase_count = target_rate / std::gcd(source_rate, target_rate);
    if (phase_count <= kMaxPrecomputedPhases) {
        return resample_polyphase_range(samples, n_samples, source_rate, target_rate, range_start,
                                        range_count, cutoff, half_taps);
    }

    std::vector<float> resampled(range_count);
    for (int32_t local_index = 0; local_index < range_count; ++local_index) {
        const int32_t i = range_start + local_index;
        const double src_pos = static_cast<double>(i) * static_cast<double>(source_rate) /
                               static_cast<double>(target_rate);
        resampled[static_cast<std::size_t>(local_index)] =
            resample_at_position(samples, n_samples, src_pos, cutoff, half_taps);
    }
    return resampled;
}

} // namespace trtmc
