/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/layoutlmv3/runtime/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace trtmc::layoutlmv3 {
namespace {
struct SpanWeights {
    int32_t first;
    std::vector<float> weights;
};
std::vector<SpanWeights> plan(int32_t source, int32_t target) {
    const float scale = static_cast<float>(source) / target;
    const float support = std::max(scale, 1.0F), inverse = scale >= 1 ? 1.0 / scale : 1.0;
    std::vector<SpanWeights> result(target);
    for (int32_t i = 0; i < target; ++i) {
        const float center = scale * (i + 0.5);
        const int32_t first = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        const int32_t end = std::min(static_cast<int32_t>(center + support + 0.5), source);
        auto& span = result[i];
        span.first = first;
        float total = 0;
        for (int32_t j = first; j < end; ++j) {
            const float value = std::max(0.0F, 1.0F - std::abs((j - center + 0.5F) * inverse));
            span.weights.push_back(value);
            total += value;
        }
        if (total <= 0)
            throw std::runtime_error("LayoutLMv3 resize has empty support");
        for (auto& value : span.weights)
            value /= total;
    }
    return result;
}
} // namespace
std::vector<float> image_pixels(const internal::ImageView& image) {
    if (!image.data || !image.height || !image.width || image.channels != 3 ||
        image.height > INT32_MAX || image.width > INT32_MAX)
        throw std::invalid_argument("LayoutLMv3 requires contiguous RGB pixels");
    const bool byte = image.format == internal::ImageFormat::UInt8;
    if ((!byte && image.format != internal::ImageFormat::Float32) ||
        image.byte_size !=
            static_cast<uint64_t>(image.height) * image.width * 3 * (byte ? 1 : sizeof(float)))
        throw std::invalid_argument("Invalid LayoutLMv3 RGB storage");
    const auto xp = plan(image.width, 224), yp = plan(image.height, 224);
    std::vector<float> horizontal(static_cast<size_t>(image.height) * 224 * 3);
    for (uint32_t y = 0; y < image.height; ++y)
        for (uint32_t x = 0; x < 224; ++x)
            for (uint32_t c = 0; c < 3; ++c) {
                const auto& span = xp[x];
                float sum = 0;
                for (size_t k = 0; k < span.weights.size(); ++k) {
                    const auto index =
                        (static_cast<size_t>(y) * image.width + span.first + k) * 3 + c;
                    const float value = byte
                                            ? static_cast<const uint8_t*>(image.data)[index]
                                            : static_cast<const float*>(image.data)[index] * 255.0F;
                    if (!std::isfinite(value) || value < 0 || value > 255)
                        throw std::invalid_argument("Invalid LayoutLMv3 RGB value");
                    sum += value * span.weights[k];
                }
                horizontal[(static_cast<size_t>(y) * 224 + x) * 3 + c] = sum;
            }
    std::vector<float> output(3 * 224 * 224);
    for (uint32_t y = 0; y < 224; ++y)
        for (uint32_t x = 0; x < 224; ++x)
            for (uint32_t c = 0; c < 3; ++c) {
                const auto& span = yp[y];
                float sum = 0;
                for (size_t k = 0; k < span.weights.size(); ++k)
                    sum += horizontal[((span.first + k) * 224 + x) * 3 + c] * span.weights[k];
                output[(static_cast<size_t>(c) * 224 + y) * 224 + x] = (sum - 127.5F) / 127.5F;
            }
    return output;
}
} // namespace trtmc::layoutlmv3
