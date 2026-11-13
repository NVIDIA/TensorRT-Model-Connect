/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The canary resampler against its per-output definition: every output is the
// Hann-windowed sinc of its in-range taps, accumulated tap by tap and divided
// by the sum of those taps' weights. The production code tabulates the
// weights per phase and filters several outputs per pass; its outputs must
// equal the definition's bit for bit.

#include "families/canary/runtime/resampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

constexpr std::int32_t kHalfTaps = 16;
constexpr double kPi = 3.14159265358979323846;

double reference_sinc(double distance, double cutoff) {
    if (std::abs(distance) < 1.0e-12)
        return cutoff;
    return cutoff * std::sin(kPi * distance * cutoff) / (kPi * distance * cutoff);
}

double reference_hann(double distance) {
    const double position = (distance + static_cast<double>(kHalfTaps)) / (2.0 * kHalfTaps);
    return 0.5 * (1.0 - std::cos(2.0 * kPi * position));
}

// The weight of tap `offset` (-kHalfTaps < offset <= kHalfTaps) at a source
// position `fraction` past the output's center sample.
double reference_weight(std::int32_t offset, double fraction, double cutoff) {
    const auto distance = static_cast<double>(offset) - fraction;
    return reference_sinc(distance, cutoff) * reference_hann(distance);
}

// One output: its center sample and phase from the exact rational position,
// then its in-range taps accumulated in order. `weights` caches each phase's
// 2 * kHalfTaps weights (offsets -kHalfTaps + 1 .. kHalfTaps).
float reference_output(const std::vector<float>& samples, std::int32_t source_rate,
                       std::int32_t target_rate, std::int32_t index, double cutoff,
                       std::vector<std::vector<double>>& weights) {
    const std::int32_t rate_gcd = std::gcd(source_rate, target_rate);
    const auto numerator = static_cast<std::int64_t>(index) * source_rate;
    const auto center = static_cast<std::int32_t>(numerator / target_rate);
    const auto phase = static_cast<std::int32_t>(numerator % target_rate) / rate_gcd;
    auto& phase_weights = weights[static_cast<std::size_t>(phase)];
    if (phase_weights.empty()) {
        const auto fraction =
            static_cast<double>(phase * rate_gcd) / static_cast<double>(target_rate);
        for (std::int32_t offset = -kHalfTaps + 1; offset <= kHalfTaps; ++offset)
            phase_weights.push_back(reference_weight(offset, fraction, cutoff));
    }
    const auto num_samples = static_cast<std::int32_t>(samples.size());
    const auto first = std::max(-kHalfTaps + 1, -center);
    const auto last = std::min(kHalfTaps, num_samples - 1 - center);
    double value = 0.0;
    double weight_sum = 0.0;
    for (std::int32_t offset = first; offset <= last; ++offset) {
        const auto weight = phase_weights[static_cast<std::size_t>(offset + kHalfTaps - 1)];
        value += static_cast<double>(samples[static_cast<std::size_t>(center + offset)]) * weight;
        weight_sum += weight;
    }
    return weight_sum > 1.0e-12 ? static_cast<float>(value / weight_sum) : 0.0F;
}

std::vector<float> reference_resample_range(const std::vector<float>& samples,
                                            std::int32_t source_rate, std::int32_t target_rate,
                                            std::int32_t output_start, std::int32_t output_count) {
    const auto num_samples = static_cast<std::int32_t>(samples.size());
    const auto full_length = static_cast<std::int32_t>(static_cast<std::int64_t>(num_samples) *
                                                       target_rate / source_rate);
    const auto start = std::clamp(output_start, 0, full_length);
    const auto count = std::clamp(output_count, 0, full_length - start);
    const double cutoff =
        std::min(1.0, static_cast<double>(target_rate) / static_cast<double>(source_rate));
    std::vector<std::vector<double>> weights(
        static_cast<std::size_t>(target_rate / std::gcd(source_rate, target_rate)));
    std::vector<float> output(static_cast<std::size_t>(count));
    for (std::int32_t local = 0; local < count; ++local) {
        output[static_cast<std::size_t>(local)] =
            reference_output(samples, source_rate, target_rate, start + local, cutoff, weights);
    }
    return output;
}

bool same_bits(const std::vector<float>& actual, const std::vector<float>& expected) {
    return actual.size() == expected.size() &&
           (actual.empty() ||
            std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0);
}

std::vector<float> seeded_audio(std::int32_t length, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> noise(0.0F, 0.25F);
    std::vector<float> samples(static_cast<std::size_t>(length));
    for (std::int32_t i = 0; i < length; ++i) {
        samples[static_cast<std::size_t>(i)] =
            static_cast<float>(0.4 * std::sin(0.013 * i) + 0.2 * std::cos(0.0021 * i)) +
            noise(generator);
    }
    // Leading silence: exact zeros through the same taps.
    std::fill_n(samples.begin(), length / 9, 0.0F);
    return samples;
}

void test_resampler_matches_per_output_definition() {
    const std::int32_t source_rates[] = {48000, 44100, 96000, 22050, 32000, 11025, 8000, 24000};
    const std::int32_t lengths[] = {1, 2, 5, 31, 32, 33, 34, 35, 64, 97, 4801, 48000, 199680};
    std::uint32_t seed = 7;
    for (const std::int32_t source_rate : source_rates) {
        for (const std::int32_t length : lengths) {
            const auto samples = seeded_audio(length, seed++);
            const auto full = trtmc::resample_linear(samples.data(), length, source_rate, 16000);
            const auto full_length =
                static_cast<std::int32_t>(static_cast<std::int64_t>(length) * 16000 / source_rate);
            const std::string name = "canary resampler matches the per-output definition at " +
                                     std::to_string(source_rate) + " Hz, " +
                                     std::to_string(length) + " samples";
            check(same_bits(full,
                            reference_resample_range(samples, source_rate, 16000, 0, full_length)),
                  name);

            // Ranges that start and end inside, at, and past both edges.
            const std::int32_t starts[] = {0, 1, 3, full_length / 3, full_length - 2};
            for (const std::int32_t start : starts) {
                for (const std::int32_t count : {1, 3, 4, 5, 9, 70, full_length - start + 1}) {
                    const auto range = trtmc::resample_linear_range(
                        samples.data(), length, source_rate, 16000, start, count);
                    check(same_bits(range, reference_resample_range(samples, source_rate, 16000,
                                                                    start, count)),
                          name + ", range " + std::to_string(start) + "+" + std::to_string(count));
                }
            }
        }
    }
}

void test_resampler_upsampling_and_identity() {
    const auto samples = seeded_audio(3001, 99);
    const auto up = trtmc::resample_linear(samples.data(), 3001, 8000, 16000);
    check(same_bits(up, reference_resample_range(samples, 8000, 16000, 0, 6002)),
          "canary resampler upsampling matches the per-output definition");

    const auto same = trtmc::resample_linear(samples.data(), 3001, 16000, 16000);
    check(same_bits(same, samples), "canary resampler passes a matching rate through");
}

// Interior outputs whose weighted sum nearly cancels (the center sample is
// solved against its 31 neighbours), so a double-level rounding difference in
// the accumulation reaches the float output.
void test_resampler_cancellation_inputs() {
    for (const std::int32_t source_rate : {48000, 44100, 24000, 8000}) {
        const std::int32_t n = 60000;
        auto samples = seeded_audio(n, 1234);
        const std::int32_t rate_gcd = std::gcd(source_rate, 16000);
        const double cutoff = std::min(1.0, 16000.0 / source_rate);
        const auto full =
            static_cast<std::int32_t>(static_cast<std::int64_t>(n) * 16000 / source_rate);
        std::int32_t next_free = 0;
        for (std::int32_t index = 0; index < full; ++index) {
            const auto numerator = static_cast<std::int64_t>(index) * source_rate;
            const auto center = static_cast<std::int32_t>(numerator / 16000);
            const auto phase = static_cast<std::int32_t>(numerator % 16000) / rate_gcd;
            const double fraction = static_cast<double>(phase * rate_gcd) / 16000.0;
            if (center - kHalfTaps + 1 < next_free || center + kHalfTaps >= n)
                continue;
            double rest = 0.0;
            for (std::int32_t offset = -kHalfTaps + 1; offset <= kHalfTaps; ++offset) {
                if (offset != 0)
                    rest += samples[static_cast<std::size_t>(center + offset)] *
                            reference_weight(offset, fraction, cutoff);
            }
            samples[static_cast<std::size_t>(center)] =
                static_cast<float>(-rest / reference_weight(0, fraction, cutoff));
            next_free = center + kHalfTaps + 1;
        }
        check(same_bits(trtmc::resample_linear(samples.data(), n, source_rate, 16000),
                        reference_resample_range(samples, source_rate, 16000, 0, full)),
              "canary resampler matches the definition on cancelling sums at " +
                  std::to_string(source_rate) + " Hz");
    }
}

} // namespace

int main() {
    test_resampler_matches_per_output_definition();
    test_resampler_upsampling_and_identity();
    test_resampler_cancellation_inputs();
    return failures;
}
