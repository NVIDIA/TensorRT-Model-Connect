/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_mask_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc::sam2 {

namespace {

std::size_t checkedArea(std::int32_t height, std::int32_t width) {
    if (height <= 0 || width <= 0)
        throw std::invalid_argument("SAM2 mask dimensions must be positive");
    const auto h = static_cast<std::size_t>(height);
    const auto w = static_cast<std::size_t>(width);
    if (w > std::numeric_limits<std::size_t>::max() / h)
        throw std::overflow_error("SAM2 mask area overflows");
    return h * w;
}

struct AxisSample {
    std::int32_t low;
    std::int32_t high;
    float high_weight;
};

std::vector<AxisSample> makeAxis(std::int32_t input_size, std::int32_t output_size) {
    std::vector<AxisSample> result(static_cast<std::size_t>(output_size));
    const float scale = static_cast<float>(input_size) / static_cast<float>(output_size);
    for (std::int32_t output = 0; output < output_size; ++output) {
        const float source = (static_cast<float>(output) + 0.5F) * scale - 0.5F;
        const auto floor_source = static_cast<std::int32_t>(std::floor(source));
        const auto low = std::clamp(floor_source, 0, input_size - 1);
        const auto high = std::clamp(floor_source + 1, 0, input_size - 1);
        float weight = source - static_cast<float>(floor_source);
        if (low == high)
            weight = 0.0F;
        result[static_cast<std::size_t>(output)] = {low, high, weight};
    }
    return result;
}

float separatelyRoundedLinear(float low, float high, float high_weight) noexcept {
    volatile float difference = high - low;
    volatile float scaled = difference * high_weight;
    volatile float result = low + scaled;
    return result;
}

} // namespace

std::vector<std::uint8_t> resizeAndThresholdMask(const float* mask_logits,
                                                 std::int32_t source_height,
                                                 std::int32_t source_width,
                                                 std::int32_t output_height,
                                                 std::int32_t output_width) {
    const auto source_area = checkedArea(source_height, source_width);
    const auto output_area = checkedArea(output_height, output_width);
    if (mask_logits == nullptr)
        throw std::invalid_argument("SAM2 mask logits must not be null");
    for (std::size_t index = 0; index < source_area; ++index) {
        if (!std::isfinite(mask_logits[index]))
            throw std::invalid_argument("SAM2 mask logits must be finite");
    }

    const auto y_axis = makeAxis(source_height, output_height);
    const auto x_axis = makeAxis(source_width, output_width);
    std::vector<std::uint8_t> result(output_area);
    for (std::int32_t y = 0; y < output_height; ++y) {
        const auto& ys = y_axis[static_cast<std::size_t>(y)];
        for (std::int32_t x = 0; x < output_width; ++x) {
            const auto& xs = x_axis[static_cast<std::size_t>(x)];
            const float top_left =
                mask_logits[static_cast<std::size_t>(ys.low) * source_width + xs.low];
            const float top_right =
                mask_logits[static_cast<std::size_t>(ys.low) * source_width + xs.high];
            const float bottom_left =
                mask_logits[static_cast<std::size_t>(ys.high) * source_width + xs.low];
            const float bottom_right =
                mask_logits[static_cast<std::size_t>(ys.high) * source_width + xs.high];
            // PyTorch's scalar formulation performs the x interpolation first.
            const float top = separatelyRoundedLinear(top_left, top_right, xs.high_weight);
            const float bottom = separatelyRoundedLinear(bottom_left, bottom_right, xs.high_weight);
            const float value = separatelyRoundedLinear(top, bottom, ys.high_weight);
            if (!std::isfinite(value))
                throw std::runtime_error("SAM2 mask resize produced a non-finite value");
            result[static_cast<std::size_t>(y) * output_width + x] =
                static_cast<std::uint8_t>(value > 0.0F);
        }
    }
    return result;
}

} // namespace trtmc::sam2
