/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/canary/runtime/resampler.h"

#include "families/canary/runtime/canary_parallel.h"

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
    // Per phase, its taps summed one by one from 0.0: the weight sum of an
    // output whose taps all fall inside the signal, in its own order.
    std::vector<double> full_sums;
};

PhaseTable make_phase_table(std::int32_t source_rate, std::int32_t target_rate, double cutoff,
                            std::int32_t half_taps) {
    PhaseTable table;
    table.rate_gcd = std::gcd(source_rate, target_rate);
    table.phase_count = target_rate / table.rate_gcd;
    table.half_taps = half_taps;
    const auto tap_count = 2 * half_taps;
    table.weights.resize(static_cast<std::size_t>(table.phase_count) * tap_count);
    table.full_sums.resize(static_cast<std::size_t>(table.phase_count));
    for (std::int32_t phase = 0; phase < table.phase_count; ++phase) {
        const auto fraction =
            static_cast<double>(phase * table.rate_gcd) / static_cast<double>(target_rate);
        double weight_sum = 0.0;
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const auto offset = tap - half_taps + 1;
            const auto distance = static_cast<double>(offset) - fraction;
            const auto weight = scaled_sinc(distance, cutoff) * hann_window(distance, half_taps);
            table.weights[static_cast<std::size_t>(phase) * tap_count + tap] = weight;
            weight_sum += weight;
        }
        table.full_sums[static_cast<std::size_t>(phase)] = weight_sum;
    }
    return table;
}

struct PhaseTap {
    std::int32_t center;
    const double* weights;
    double full_sum;
};

// The phase table entry of consecutive outputs. Output `index` sits at source
// position index * source_rate / target_rate = center + phase / phase_count;
// stepping one output adds the same integer quotient and remainder each time,
// so the walk visits exactly the (center, phase) pairs the division gives.
class PhaseCursor {
  public:
    PhaseCursor(const PhaseTable& table, std::int32_t index, std::int32_t source_rate,
                std::int32_t target_rate)
        : table_(&table), step_center_(source_rate / target_rate),
          step_phase_((source_rate % target_rate) / table.rate_gcd) {
        const auto numerator = static_cast<std::int64_t>(index) * source_rate;
        center_ = static_cast<std::int32_t>(numerator / target_rate);
        phase_ = static_cast<std::int32_t>(numerator % target_rate) / table.rate_gcd;
    }

    PhaseTap tap() const {
        return {center_,
                table_->weights.data() + static_cast<std::size_t>(phase_) * 2 * table_->half_taps,
                table_->full_sums[static_cast<std::size_t>(phase_)]};
    }

    void advance() {
        center_ += step_center_;
        phase_ += step_phase_;
        if (phase_ >= table_->phase_count) {
            phase_ -= table_->phase_count;
            ++center_;
        }
    }

  private:
    const PhaseTable* table_;
    std::int32_t step_center_;
    std::int32_t step_phase_;
    std::int32_t center_{0};
    std::int32_t phase_{0};
};

float normalized(double value, double weight_sum) {
    return weight_sum > 1.0e-12 ? static_cast<float>(value / weight_sum) : 0.0F;
}

// One output near either end of the signal: only its in-range taps count.
float resample_edge(const float* samples, std::int32_t num_samples, const PhaseTap& tap,
                    std::int32_t half_taps) {
    const auto first = std::max(-half_taps + 1, -tap.center);
    const auto last = std::min(half_taps, num_samples - 1 - tap.center);
    double value = 0.0;
    double weight_sum = 0.0;
    for (std::int32_t offset = first; offset <= last; ++offset) {
        const auto weight = tap.weights[offset + half_taps - 1];
        value += static_cast<double>(samples[tap.center + offset]) * weight;
        weight_sum += weight;
    }
    return normalized(value, weight_sum);
}

// Interior outputs are filtered kLanes at a time: each lane accumulates its
// own taps in tap order, exactly as one output alone would, and the
// independent lanes let the core overlap their multiply-add chains.
constexpr std::int32_t kLanes = 4;

// The outputs of one resample_linear_range call.
struct WalkPlan {
    std::int32_t num_samples;
    std::int32_t source_rate;
    std::int32_t target_rate;
    std::int32_t output_start;
    std::int32_t output_count;
};

// Outputs [begin, end) of the walk over the plan's outputs. The walk takes an
// output alone while it lies near either end of the signal and kLanes outputs
// at a time in between. `begin` and `end` must be outputs the walk from 0
// starts a step at, or the count (walk_ranges), so that this call takes the
// same steps over [begin, end) as the walk from 0 does and writes no other
// output.
void resample_walk(const PhaseTable& table, const float* samples, const WalkPlan& plan,
                   std::int32_t begin, std::int32_t end, float* output) {
    const auto half_taps = table.half_taps;
    const auto tap_count = 2 * half_taps;
    // An output reads all its taps when its center lies in [lowest, highest].
    const auto lowest = half_taps - 1;
    const auto highest = plan.num_samples - 1 - half_taps;

    PhaseCursor cursor(table, plan.output_start + begin, plan.source_rate, plan.target_rate);
    std::int32_t local = begin;
    while (local < end) {
        if (local + kLanes <= plan.output_count) {
            PhaseCursor probe = cursor;
            PhaseTap taps[kLanes];
            bool interior = true;
            for (std::int32_t lane = 0; lane < kLanes; ++lane) {
                taps[lane] = probe.tap();
                interior = interior && taps[lane].center >= lowest && taps[lane].center <= highest;
                probe.advance();
            }
            if (interior) {
                double values[kLanes] = {};
                for (std::int32_t tap = 0; tap < tap_count; ++tap) {
                    for (std::int32_t lane = 0; lane < kLanes; ++lane) {
                        const float* window = samples + (taps[lane].center - half_taps + 1);
                        values[lane] += static_cast<double>(window[tap]) * taps[lane].weights[tap];
                    }
                }
                for (std::int32_t lane = 0; lane < kLanes; ++lane)
                    output[static_cast<std::size_t>(local + lane)] =
                        normalized(values[lane], taps[lane].full_sum);
                cursor = probe;
                local += kLanes;
                continue;
            }
        }
        output[static_cast<std::size_t>(local)] =
            resample_edge(samples, plan.num_samples, cursor.tap(), half_taps);
        cursor.advance();
        ++local;
    }
}

// The ranges of outputs the walk is cut into. Output centers never decrease,
// so the walk from 0 takes its first `head` outputs alone (their centers lie
// below the first interior one), then kLanes at a time, then, from the first
// group that does not fit the interior or the count, alone again to the end.
// A cut `head + kLanes * k` is therefore always an output the walk starts a
// step at: every output is computed in the same lane of the same group, or
// alone, however the walk is cut, and a range writes its own outputs only.
std::vector<canary::Range> walk_ranges(const WalkPlan& plan, std::int32_t half_taps,
                                       std::int32_t workers, std::int32_t min_outputs_per_range) {
    std::vector<canary::Range> ranges;
    if (plan.output_count <= 0)
        return ranges;
    // Output `index` has center index * source_rate / target_rate (rounded
    // down): the first one at or above `lowest`.
    const auto lowest = static_cast<std::int64_t>(half_taps - 1);
    const auto first_interior =
        (lowest * plan.target_rate + plan.source_rate - 1) / plan.source_rate;
    const auto head =
        std::clamp<std::int64_t>(first_interior - plan.output_start, 0, plan.output_count);
    const std::int64_t groups = (plan.output_count - head + kLanes - 1) / kLanes;
    const auto min_groups =
        (std::max<std::int64_t>(min_outputs_per_range, 1) + kLanes - 1) / kLanes;
    const auto group_ranges = canary::pass_ranges(groups, min_groups, workers);
    if (group_ranges.empty()) {
        ranges.emplace_back(0, plan.output_count);
        return ranges;
    }
    ranges.reserve(group_ranges.size());
    for (const auto& [first, last] : group_ranges) {
        ranges.emplace_back(first == 0 ? 0 : head + first * kLanes,
                            std::min<std::int64_t>(plan.output_count, head + last * kLanes));
    }
    return ranges;
}

std::vector<float> resample_phases(const float* samples, const WalkPlan& plan, double cutoff,
                                   std::int32_t half_taps, canary::RangeRunner* runner,
                                   std::int32_t min_outputs_per_range) {
    const auto table = make_phase_table(plan.source_rate, plan.target_rate, cutoff, half_taps);
    std::vector<float> output(static_cast<std::size_t>(plan.output_count));
    const std::int32_t workers = runner != nullptr ? runner->workers() : 1;
    canary::run_pass(walk_ranges(plan, half_taps, workers, min_outputs_per_range), runner,
                     [&](std::int64_t begin, std::int64_t end, std::int32_t) {
                         resample_walk(table, samples, plan, static_cast<std::int32_t>(begin),
                                       static_cast<std::int32_t>(end), output.data());
                     });
    return output;
}

} // namespace

std::vector<float> resample_linear(const float* samples, std::int32_t num_samples,
                                   std::int32_t source_rate, std::int32_t target_rate,
                                   canary::RangeRunner* runner,
                                   std::int32_t min_outputs_per_range) {
    if (num_samples <= 0)
        return {};
    return resample_linear_range(samples, num_samples, source_rate, target_rate, 0,
                                 output_length(num_samples, source_rate, target_rate), runner,
                                 min_outputs_per_range);
}

std::vector<float> resample_linear_range(const float* samples, std::int32_t num_samples,
                                         std::int32_t source_rate, std::int32_t target_rate,
                                         std::int32_t output_start, std::int32_t output_count,
                                         canary::RangeRunner* runner,
                                         std::int32_t min_outputs_per_range) {
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
        return resample_phases(samples, {num_samples, source_rate, target_rate, start, count},
                               cutoff, half_taps, runner, min_outputs_per_range);
    std::vector<float> output(static_cast<std::size_t>(count));
    const std::int32_t workers = runner != nullptr ? runner->workers() : 1;
    canary::run_pass(canary::pass_ranges(count, min_outputs_per_range, workers), runner,
                     [&](std::int64_t begin, std::int64_t end, std::int32_t) {
                         for (auto local = static_cast<std::int32_t>(begin);
                              local < static_cast<std::int32_t>(end); ++local) {
                             const auto position =
                                 static_cast<double>(start + local) * source_rate / target_rate;
                             output[static_cast<std::size_t>(local)] =
                                 resample_at(samples, num_samples, position, cutoff, half_taps);
                         }
                     });
    return output;
}

} // namespace trtmc
