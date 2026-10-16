/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/personaplex/runtime/resampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace trtmc {
namespace {

// The Native reference loads audio with librosa.resample, whose default is
// soxr's "HQ" recipe: a linear-phase low-pass whose passband ends at 91.3% of
// the lower Nyquist frequency, with about 20-bit (126 dB) stopband rejection.
// A Kaiser-windowed sinc with the same band edges reproduces it to ~1e-5 on
// speech. Mimi tokenizes the resampled waveform, so a looser filter changes
// the codes the model is conditioned on.
constexpr double kPassbandEnd = 0.913;
constexpr double kStopbandAttenuationDb = 126.0;
constexpr double kPi = 3.14159265358979323846;

struct KaiserLowpass {
    double cutoff{0.5}; // cycles per source sample
    double beta{0.0};
    std::int32_t half_taps{1};
};

KaiserLowpass design_lowpass(std::int32_t source_rate, std::int32_t target_rate) {
    const double nyquist = 0.5 * static_cast<double>(std::min(source_rate, target_rate));
    const double cutoff_hz = 0.5 * nyquist * (1.0 + kPassbandEnd);
    const double transition = nyquist * (1.0 - kPassbandEnd) / static_cast<double>(source_rate);
    const double taps = (kStopbandAttenuationDb - 7.95) / (2.285 * 2.0 * kPi * transition);
    KaiserLowpass filter;
    filter.cutoff = cutoff_hz / static_cast<double>(source_rate);
    filter.beta = 0.1102 * (kStopbandAttenuationDb - 8.7);
    filter.half_taps = static_cast<std::int32_t>(std::ceil(0.5 * taps)) + 1;
    return filter;
}

// Modified Bessel function of the first kind, order zero (power series).
double bessel_i0(double x) {
    const double quarter_square = 0.25 * x * x;
    double term = 1.0;
    double sum = 1.0;
    for (int k = 1; k < 200 && term > 1.0e-17 * sum; ++k) {
        term *= quarter_square / (static_cast<double>(k) * static_cast<double>(k));
        sum += term;
    }
    return sum;
}

double kernel(double distance, const KaiserLowpass& filter, double window_norm) {
    const double ratio = distance / static_cast<double>(filter.half_taps);
    if (std::abs(ratio) >= 1.0)
        return 0.0;
    const double window = bessel_i0(filter.beta * std::sqrt(1.0 - ratio * ratio)) / window_norm;
    const double argument = 2.0 * filter.cutoff * distance;
    const double sinc =
        std::abs(argument) < 1.0e-12 ? 1.0 : std::sin(kPi * argument) / (kPi * argument);
    return 2.0 * filter.cutoff * sinc * window;
}

} // namespace

std::vector<float> resample_audio(const float* samples, std::int32_t num_samples,
                                  std::int32_t source_rate, std::int32_t target_rate) {
    if (samples == nullptr || num_samples < 0 || source_rate <= 0 || target_rate <= 0)
        throw std::invalid_argument("resampler requires samples and positive rates");
    if (source_rate == target_rate)
        return {samples, samples + num_samples};
    if (num_samples == 0)
        return {};

    // librosa sizes the output as ceil(samples * target / source).
    const auto output_count = static_cast<std::int32_t>(
        (static_cast<std::int64_t>(num_samples) * target_rate + source_rate - 1) / source_rate);
    const auto filter = design_lowpass(source_rate, target_rate);
    const double window_norm = bessel_i0(filter.beta);
    const std::int32_t rate_gcd = std::gcd(source_rate, target_rate);
    const std::int32_t phase_count = target_rate / rate_gcd;
    const std::int32_t tap_count = 2 * filter.half_taps;

    // Output sample k sits at source position k * source / target; its taps
    // are the source samples base - half + 1 .. base + half, zero outside.
    std::vector<double> weights(static_cast<std::size_t>(phase_count) * tap_count);
    for (std::int32_t phase = 0; phase < phase_count; ++phase) {
        const double fraction = static_cast<double>(phase) * rate_gcd / target_rate;
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const double offset = static_cast<double>(tap - filter.half_taps + 1);
            weights[static_cast<std::size_t>(phase) * tap_count + tap] =
                kernel(offset - fraction, filter, window_norm);
        }
    }

    std::vector<float> output(static_cast<std::size_t>(output_count));
    for (std::int32_t index = 0; index < output_count; ++index) {
        const auto numerator = static_cast<std::int64_t>(index) * source_rate;
        const auto base = static_cast<std::int32_t>(numerator / target_rate);
        const auto phase = static_cast<std::int32_t>(numerator % target_rate) / rate_gcd;
        const double* row = weights.data() + static_cast<std::size_t>(phase) * tap_count;
        const auto first = std::max(0, filter.half_taps - 1 - base);
        const auto last = std::min(tap_count, num_samples + filter.half_taps - 1 - base);
        double value = 0.0;
        for (std::int32_t tap = first; tap < last; ++tap)
            value += static_cast<double>(samples[base + tap - filter.half_taps + 1]) * row[tap];
        output[static_cast<std::size_t>(index)] = static_cast<float>(value);
    }
    return output;
}

} // namespace trtmc
