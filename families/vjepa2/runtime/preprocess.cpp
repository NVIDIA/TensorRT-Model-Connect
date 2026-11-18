/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vjepa2/runtime/preprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace trtmc::vjepa2 {
namespace {
struct SpanWeights {
    int32_t first;
    std::vector<float> values;
};
std::vector<SpanWeights> bilinear_plan(int32_t source, int32_t target, int32_t offset,
                                       int32_t count) {
    // The float input path of PyTorch's separable antialiased bilinear resize:
    // triangular filter, half-pixel centers and renormalized boundary support.
    const float scale = static_cast<float>(source) / target;
    const float support = std::max(scale, 1.0F), inverse = scale >= 1 ? 1.0 / scale : 1.0;
    std::vector<SpanWeights> result(count);
    for (int32_t i = 0; i < count; ++i) {
        const float center = scale * (offset + i + 0.5);
        const int32_t first = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        const int32_t end = std::min(static_cast<int32_t>(center + support + 0.5), source);
        auto& span = result[i];
        span.first = first;
        float total = 0;
        for (int32_t j = first; j < end; ++j) {
            const float distance = (j - center + 0.5) * inverse;
            const float value = std::max(0.0F, 1.0F - std::abs(distance));
            span.values.push_back(value);
            total += value;
        }
        if (total <= 0)
            throw std::runtime_error("V-JEPA2 resize has empty support");
        for (auto& value : span.values)
            value /= total;
    }
    return result;
}
int32_t crop_offset(int32_t difference) {
    // Python round used by torchvision center_crop rounds half to even.
    const int32_t half = difference / 2;
    return half + ((difference % 2 && half % 2) ? 1 : 0);
}
} // namespace
ImageGeometry image_geometry(int32_t h, int32_t w, int32_t shortest, int32_t crop) {
    if (h <= 0 || w <= 0 || shortest < crop || crop <= 0)
        throw std::invalid_argument("Invalid V-JEPA2 frame or resize dimensions");
    const int64_t rh = h <= w ? shortest : static_cast<int64_t>(shortest) * h / w;
    const int64_t rw = w <= h ? shortest : static_cast<int64_t>(shortest) * w / h;
    if (rh > INT32_MAX || rw > INT32_MAX)
        throw std::invalid_argument("V-JEPA2 resized dimensions overflow");
    return {static_cast<int32_t>(rh), static_cast<int32_t>(rw),
            crop_offset(static_cast<int32_t>(rh) - crop),
            crop_offset(static_cast<int32_t>(rw) - crop)};
}
std::vector<float> preprocess(const float* rgb, int32_t h, int32_t w, int32_t shortest,
                              int32_t crop) {
    if (!rgb)
        throw std::invalid_argument("V-JEPA2 requires RGB pixels");
    const auto geometry = image_geometry(h, w, shortest, crop);
    const auto xp = bilinear_plan(w, geometry.width, geometry.left, crop);
    const auto yp = bilinear_plan(h, geometry.height, geometry.top, crop);
    std::vector<float> horizontal(static_cast<std::size_t>(h) * crop * 3);
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < crop; ++x) {
            const auto& span = xp[x];
            for (int32_t c = 0; c < 3; ++c) {
                float value = 0;
                for (std::size_t k = 0; k < span.values.size(); ++k) {
                    const float source =
                        rgb[(static_cast<std::size_t>(y) * w + span.first + k) * 3 + c];
                    if (!std::isfinite(source))
                        throw std::invalid_argument("V-JEPA2 frame contains nonfinite RGB");
                    value += (source * 255.0F) * span.values[k];
                }
                horizontal[(static_cast<std::size_t>(y) * crop + x) * 3 + c] = value;
            }
        }
    }
    const float mean[]{0.485F * 255.0F, 0.456F * 255.0F, 0.406F * 255.0F};
    const float deviation[]{0.229F * 255.0F, 0.224F * 255.0F, 0.225F * 255.0F};
    const auto plane = static_cast<std::size_t>(crop) * crop;
    std::vector<float> output(3 * plane);
    for (int32_t y = 0; y < crop; ++y) {
        const auto& span = yp[y];
        for (int32_t x = 0; x < crop; ++x) {
            for (int32_t c = 0; c < 3; ++c) {
                float value = 0;
                for (std::size_t k = 0; k < span.values.size(); ++k)
                    value += horizontal[((span.first + k) * crop + x) * 3 + c] * span.values[k];
                output[c * plane + y * crop + x] = (value - mean[c]) / deviation[c];
            }
        }
    }
    return output;
}
} // namespace trtmc::vjepa2
