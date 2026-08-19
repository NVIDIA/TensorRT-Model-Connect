/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc::sam2 {

namespace {

constexpr std::int64_t kPillowCoefficientScale = std::int64_t{1} << kPillowResizePrecisionBits;

double bicubic(double value) {
    constexpr double kA = -0.5;
    value = std::abs(value);
    if (value < 1.0)
        return ((kA + 2.0) * value - (kA + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return ((kA * value - 5.0 * kA) * value + 8.0 * kA) * value - 4.0 * kA;
    return 0.0;
}

std::vector<double> makeFloatingWeights(std::int32_t first, std::int32_t end, double center,
                                        double inverse_filter_scale, double& sum) {
    std::vector<double> result(static_cast<std::size_t>(end - first));
    sum = 0.0;
    for (std::int32_t input_index = first; input_index < end; ++input_index) {
        const double distance =
            (static_cast<double>(input_index) - center + 0.5) * inverse_filter_scale;
        const double weight = bicubic(distance);
        result[static_cast<std::size_t>(input_index - first)] = weight;
        sum += weight;
    }
    return result;
}

void appendQuantizedWeights(PillowResizeAxisPlan& plan, const std::vector<double>& floating,
                            double sum) {
    plan.weights.reserve(plan.weights.size() + floating.size());
    for (const double raw_weight : floating) {
        const double scaled = raw_weight / sum * static_cast<double>(kPillowCoefficientScale);
        if (!std::isfinite(scaled) ||
            scaled < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
            scaled > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("SAM2 Pillow resize coefficient overflowed");
        }
        plan.weights.push_back(
            static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
    }
}

} // namespace

PillowResizeAxisPlan makePillowBicubicAxisPlan(std::int32_t input_size, std::int32_t output_size) {
    if (input_size <= 0 || output_size <= 0)
        throw std::invalid_argument("SAM2 resize dimensions must be positive");
    const double scale = static_cast<double>(input_size) / static_cast<double>(output_size);
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    PillowResizeAxisPlan plan;
    plan.spans.resize(static_cast<std::size_t>(output_size));
    for (std::int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const auto first =
            std::max<std::int32_t>(0, static_cast<std::int32_t>(center - support + 0.5));
        const auto end =
            std::min<std::int32_t>(input_size, static_cast<std::int32_t>(center + support + 0.5));
        if (end <= first)
            throw std::runtime_error("SAM2 Pillow resize produced empty filter support");

        auto& span = plan.spans[static_cast<std::size_t>(output_index)];
        span.first = first;
        span.weight_offset = static_cast<std::int32_t>(plan.weights.size());
        span.weight_count = end - first;
        double sum = 0.0;
        const auto floating = makeFloatingWeights(first, end, center, inverse_filter_scale, sum);
        if (!std::isfinite(sum) || sum == 0.0)
            throw std::runtime_error("SAM2 Pillow resize has invalid coefficient sum");
        appendQuantizedWeights(plan, floating, sum);
    }
    return plan;
}

const Sam2Rgb8NormalizationTable& sam2Rgb8NormalizationTable() {
    static const Sam2Rgb8NormalizationTable table = [] {
        constexpr std::array<float, kSam2RgbChannels> kMean = {0.485F, 0.456F, 0.406F};
        constexpr std::array<float, kSam2RgbChannels> kStd = {0.229F, 0.224F, 0.225F};
        Sam2Rgb8NormalizationTable result{};
        for (std::size_t channel = 0; channel < kSam2RgbChannels; ++channel) {
            for (std::size_t value = 0; value < kSam2Rgb8ValueCount; ++value) {
                const float source_value = static_cast<float>(static_cast<double>(value) / 255.0);
                result[channel * kSam2Rgb8ValueCount + value] =
                    (source_value - kMean[channel]) / kStd[channel];
            }
        }
        return result;
    }();
    return table;
}

} // namespace trtmc::sam2
