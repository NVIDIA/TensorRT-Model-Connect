/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The canary resampler against its per-output definition: every output is the
// Hann-windowed sinc of its in-range taps, accumulated tap by tap and divided
// by the sum of those taps' weights. The production code tabulates the
// weights per phase and filters several outputs per pass; its outputs must
// equal the definition's bit for bit.

#include "families/canary/runtime/canary_parallel.h"
#include "families/canary/runtime/resampler.h"
#include "families/canary/tests/cpp/canary_range_runners.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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

// The outputs (relative to output_start) at which the walk over a range's
// outputs begins a step, from the per-output definition: an output is taken
// alone unless it and the next three all read their full 2 * kHalfTaps taps,
// in which case the four are one step. The count is a step start too (the
// end of the walk).
std::vector<bool> walk_step_starts(std::int32_t num_samples, std::int32_t source_rate,
                                   std::int32_t target_rate, std::int32_t start,
                                   std::int32_t count) {
    const auto center = [&](std::int32_t local) {
        return static_cast<std::int32_t>(static_cast<std::int64_t>(start + local) * source_rate /
                                         target_rate);
    };
    const auto reads_all_taps = [&](std::int32_t local) {
        return center(local) >= kHalfTaps - 1 && center(local) <= num_samples - 1 - kHalfTaps;
    };
    std::vector<bool> starts(static_cast<std::size_t>(count) + 1, false);
    std::int32_t local = 0;
    while (local < count) {
        starts[static_cast<std::size_t>(local)] = true;
        const bool group = local + 4 <= count && reads_all_taps(local) &&
                           reads_all_taps(local + 1) && reads_all_taps(local + 2) &&
                           reads_all_taps(local + 3);
        local += group ? 4 : 1;
    }
    starts[static_cast<std::size_t>(count)] = true;
    return starts;
}

struct Rates {
    std::int32_t source;
    std::int32_t target;
};

std::string describe(const Rates& rate, std::int32_t length, std::int32_t workers,
                     std::int32_t min_outputs) {
    return std::to_string(rate.source) + " to " + std::to_string(rate.target) + " Hz, " +
           std::to_string(length) + " samples, " + std::to_string(workers) +
           " workers, shares of " + std::to_string(min_outputs);
}

// The request's signal (4.16 s of 48 kHz) goes to the runner as one pass of
// 32 ranges for eight workers, and a signal of fewer than sixteen shares does
// not go to it at all. What the runner is handed is what runs.
void test_resampler_shares_a_long_signal_only() {
    const std::int32_t length = 199680;
    const auto samples = seeded_audio(length, 20261117);
    canary_test::RecordingRunner runner(8);
    const auto shared = trtmc::resample_linear(samples.data(), length, 48000, 16000, &runner);
    check(runner.passes.size() == 1 && runner.passes[0].size() == 32 &&
              canary_test::covers_once(runner.passes[0], 66560),
          "canary resampler shares the request's 66,560 outputs as 32 ranges for eight workers");
    check(same_bits(shared, reference_resample_range(samples, 48000, 16000, 0, 66560)),
          "canary resampler's shared outputs match the per-output definition");

    canary_test::RecordingRunner two(2);
    trtmc::resample_linear_range(samples.data(), length, 48000, 16000, 100, 40000, &two);
    check(two.passes.size() == 1 && two.passes[0].size() == 8,
          "canary resampler shares a range of the outputs as 8 ranges for two workers");

    canary_test::RecordingRunner unused(8);
    const std::int32_t short_length =
        3 * (static_cast<std::int32_t>(trtmc::canary::kMinRangesToShare) *
                 trtmc::kCanaryResamplerMinOutputsPerRange -
             8);
    trtmc::resample_linear(samples.data(), short_length, 48000, 16000, &unused);
    trtmc::resample_linear(samples.data(), short_length, 16000, 16000, &unused);
    check(unused.passes.empty(),
          "canary resampler keeps fewer than sixteen shares, and an equal rate, off the runner");

    // The per-output path (too many phases for the weight table) is shared too.
    canary_test::RecordingRunner odd(4);
    trtmc::resample_linear(samples.data(), 60000, 44100, 16001, &odd, 512);
    check(odd.passes.size() == 1 &&
              canary_test::covers_once(odd.passes[0], std::int64_t{60000} * 16001 / 44100),
          "canary resampler shares the per-output path");
}

// A range must begin and end where the walk begins a step: then each output
// is filtered in the same lane of the same group, or alone, however the walk
// is cut, and no range writes another range's output.
void test_resampler_ranges_lie_on_walk_steps() {
    const Rates rates[] = {{48000, 16000}, {44100, 16000}, {96000, 16000}, {22050, 16000},
                           {11025, 16000}, {8000, 16000},  {16000, 48000}, {8000, 44100}};
    std::uint32_t seed = 31;
    for (const auto& rate : rates) {
        for (const std::int32_t length : {40, 97, 4801}) {
            const auto samples = seeded_audio(length, seed++);
            const auto full = static_cast<std::int32_t>(static_cast<std::int64_t>(length) *
                                                        rate.target / rate.source);
            for (const std::int32_t start : {0, 1, 2, 3, 5, 6, 7, full / 3, full - 9}) {
                for (const std::int32_t count : {9, 70, 1000, full}) {
                    const auto clamped_start = std::clamp(start, 0, full);
                    const auto clamped_count = std::clamp(count, 0, full - clamped_start);
                    const auto starts = walk_step_starts(length, rate.source, rate.target,
                                                         clamped_start, clamped_count);
                    for (const std::int32_t workers : {2, 8}) {
                        for (const std::int32_t min_outputs : {1, 5, 64}) {
                            canary_test::RecordingRunner runner(workers);
                            trtmc::resample_linear_range(samples.data(), length, rate.source,
                                                         rate.target, start, count, &runner,
                                                         min_outputs);
                            if (runner.passes.empty())
                                continue;
                            const auto& ranges = runner.passes[0];
                            bool on_steps = true;
                            for (const auto& [begin, end] : ranges) {
                                on_steps = on_steps && starts[static_cast<std::size_t>(begin)] &&
                                           starts[static_cast<std::size_t>(end)];
                            }
                            const std::string name = describe(rate, length, workers, min_outputs) +
                                                     ", range " + std::to_string(start) + "+" +
                                                     std::to_string(count);
                            check(runner.passes.size() == 1 &&
                                      canary_test::covers_once(ranges, clamped_count),
                                  "canary resampler ranges cover the outputs once at " + name);
                            check(on_steps,
                                  "canary resampler ranges begin and end on walk steps at " + name);
                        }
                    }
                }
            }
        }
    }
}

// One range run alone writes its own outputs, with the values of the whole
// walk, and no other output.
void test_resampler_range_writes_only_its_outputs() {
    const Rates rates[] = {{48000, 16000}, {44100, 16000}, {8000, 16000}, {44100, 16001}};
    std::uint32_t seed = 900;
    for (const auto& rate : rates) {
        const std::int32_t length = 6000;
        auto samples = seeded_audio(length, seed++);
        for (float& sample : samples)
            sample += 0.5F; // no exact zero among the outputs
        const auto whole = trtmc::resample_linear(samples.data(), length, rate.source, rate.target);
        canary_test::RecordingRunner plan(3);
        trtmc::resample_linear(samples.data(), length, rate.source, rate.target, &plan, 16);
        check(plan.passes.size() == 1 && plan.passes[0].size() == 12,
              "canary resampler cuts the test signal into 12 ranges at " +
                  describe(rate, length, 3, 16));
        if (plan.passes.size() != 1)
            continue;
        const auto ranges = plan.passes[0];
        for (std::size_t only = 0; only < ranges.size(); ++only) {
            canary_test::RecordingRunner one(
                3, [only](std::size_t, std::size_t range) { return range == only; });
            const auto part =
                trtmc::resample_linear(samples.data(), length, rate.source, rate.target, &one, 16);
            bool own = part.size() == whole.size();
            bool others_untouched = own;
            for (std::size_t index = 0; own && index < part.size(); ++index) {
                const bool inside = static_cast<std::int64_t>(index) >= ranges[only].first &&
                                    static_cast<std::int64_t>(index) < ranges[only].second;
                if (inside)
                    own = std::memcmp(&part[index], &whole[index], sizeof(float)) == 0;
                else if (part[index] != 0.0F || std::signbit(part[index]))
                    others_untouched = false;
            }
            const std::string name =
                "range " + std::to_string(only) + " at " + describe(rate, length, 3, 16);
            check(own, "canary resampler range run alone gives the whole walk's outputs: " + name);
            check(others_untouched, "canary resampler range writes no other output: " + name);
        }
    }
}

// The outputs must not depend on the runner, on its worker count or on a
// range's smallest share: a cut only changes which thread writes an output.
void test_resampler_does_not_depend_on_the_split() {
    // The last two pairs have too many phases for the weight table and take
    // the per-output path.
    const Rates rates[] = {{48000, 16000}, {44100, 16000}, {22050, 16000}, {8000, 16000},
                           {16000, 48000}, {44100, 16001}, {16001, 44100}};
    trtmc::canary::HostWorkers team(8);
    std::uint32_t seed = 4242;
    for (const auto& rate : rates) {
        // The per-output path computes every tap's weight: shorter signals.
        const bool table = 16000 / std::gcd(rate.source, 16000) <= 2048 && rate.target != 16001 &&
                           rate.target != 44100;
        for (const std::int32_t length : {97, 4801, table ? 30000 : 1500}) {
            auto samples = seeded_audio(length, seed++);
            if (length == 4801) {
                // Non-finite samples reach every output whose taps cover them.
                samples[1000] = std::numeric_limits<float>::infinity();
                samples[3000] = std::numeric_limits<float>::quiet_NaN();
            }
            const auto full = static_cast<std::int32_t>(static_cast<std::int64_t>(length) *
                                                        rate.target / rate.source);
            const auto serial =
                trtmc::resample_linear(samples.data(), length, rate.source, rate.target);
            const std::int32_t range_start = full / 3 + 1;
            const std::int32_t range_count = full / 2;
            const auto serial_range = trtmc::resample_linear_range(
                samples.data(), length, rate.source, rate.target, range_start, range_count);
            for (const std::int32_t min_outputs : {1, 5, 64, 2048}) {
                for (const std::int32_t workers : {2, 5, 8}) {
                    canary_test::RecordingRunner reversed(workers);
                    const std::string name = describe(rate, length, workers, min_outputs);
                    check(same_bits(trtmc::resample_linear(samples.data(), length, rate.source,
                                                           rate.target, &reversed, min_outputs),
                                    serial),
                          "canary resampler output does not depend on the cut or the order at " +
                              name);
                    check(same_bits(trtmc::resample_linear_range(
                                        samples.data(), length, rate.source, rate.target,
                                        range_start, range_count, &reversed, min_outputs),
                                    serial_range),
                          "canary resampler range does not depend on the cut or the order at " +
                              name);
                }
                const std::string name = describe(rate, length, team.workers(), min_outputs);
                check(same_bits(trtmc::resample_linear(samples.data(), length, rate.source,
                                                       rate.target, &team, min_outputs),
                                serial),
                      "canary resampler output on the team's threads equals one thread's at " +
                          name);
            }
        }
    }
}

} // namespace

int main() {
    test_resampler_matches_per_output_definition();
    test_resampler_upsampling_and_identity();
    test_resampler_cancellation_inputs();
    test_resampler_shares_a_long_signal_only();
    test_resampler_ranges_lie_on_walk_steps();
    test_resampler_range_writes_only_its_outputs();
    test_resampler_does_not_depend_on_the_split();
    return failures;
}
