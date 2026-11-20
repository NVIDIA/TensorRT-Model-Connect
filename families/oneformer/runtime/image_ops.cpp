/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The antialiased tap / weight computation, the int16 weight precision of the
// uint8 path and the separable interpolation loops are adapted from PyTorch
// v2.12.0, aten/src/ATen/native/cpu/UpSampleKernel.cpp
// (_compute_indices_min_size_weights_aa, _compute_index_ranges_int16_weights),
// BSD-3-Clause, which derives from Pillow's and Pillow-SIMD's Resample.c
// (MIT-CMU License). processed_size follows get_size_with_aspect_ratio in
// Hugging Face Transformers v5.2.0, src/transformers/image_transforms.py
// (Apache-2.0). The notices are reproduced in families/oneformer/NOTICE.

#include "families/oneformer/runtime/image_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc::oneformer {
namespace {

// One output position of a separable antialiased resize: its first source
// index and the weights of the consecutive source positions it reads.
template <class Weight>
struct Span {
    std::int64_t first{0};
    std::vector<Weight> weights;
};

// ATen's antialiased triangle (bilinear) plan for `input` -> `output` samples.
// `Real` is the arithmetic of the reference kernel: double for the uint8 path,
// float for float tensors (the mixed float/double expressions are kept).
template <class Real>
std::vector<Span<Real>> antialias_plan(std::int64_t input, std::int64_t output, Real& max_weight) {
    const Real scale = static_cast<Real>(input) / static_cast<Real>(output);
    const Real support = scale >= Real(1) ? scale : Real(1);
    // 1.0 / scale is a double expression in the reference, stored as Real.
    const Real invscale =
        scale >= Real(1) ? static_cast<Real>(1.0 / static_cast<double>(scale)) : Real(1);
    const auto max_size = static_cast<std::int64_t>(std::ceil(support)) * 2 + 1;
    std::vector<Span<Real>> plan(static_cast<std::size_t>(output));
    max_weight = Real(0);
    for (std::int64_t index = 0; index < output; ++index) {
        const Real center = static_cast<Real>(scale * (static_cast<double>(index) + 0.5));
        const auto first = std::max<std::int64_t>(
            static_cast<std::int64_t>(static_cast<double>(center - support) + 0.5), 0);
        auto size =
            std::min<std::int64_t>(
                static_cast<std::int64_t>(static_cast<double>(center + support) + 0.5), input) -
            first;
        size = std::clamp<std::int64_t>(size, 0, max_size);
        if (size == 0)
            throw std::runtime_error("oneformer resize produced an empty interpolation span");
        auto& span = plan[static_cast<std::size_t>(index)];
        span.first = first;
        span.weights.resize(static_cast<std::size_t>(size));
        Real total = Real(0);
        for (std::int64_t j = 0; j < size; ++j) {
            const double offset = static_cast<double>(static_cast<Real>(j + first) - center) + 0.5;
            const Real x = std::abs(static_cast<Real>(offset * static_cast<double>(invscale)));
            const Real weight = x < Real(1) ? Real(1) - x : Real(0);
            span.weights[static_cast<std::size_t>(j)] = weight;
            total += weight;
        }
        if (total != Real(0)) {
            for (auto& weight : span.weights) {
                weight /= total;
                max_weight = std::max(max_weight, weight);
            }
        }
    }
    return plan;
}

struct FixedPlan {
    std::vector<Span<std::int32_t>> spans;
    unsigned precision{0};
};

// The uint8 kernel's int16 weights: the largest precision for which the
// biggest weight still fits, then round-half-away-from-zero.
FixedPlan fixed_plan(std::int64_t input, std::int64_t output) {
    double max_weight = 0.0;
    const auto plan = antialias_plan<double>(input, output, max_weight);
    FixedPlan fixed;
    for (fixed.precision = 0; fixed.precision < 22; ++fixed.precision) {
        const auto next = static_cast<int>(0.5 + max_weight * (1 << (fixed.precision + 1)));
        if (next >= (1 << 15))
            break;
    }
    if (fixed.precision == 0)
        throw std::runtime_error("oneformer resize has no usable weight precision");
    fixed.spans.resize(plan.size());
    for (std::size_t index = 0; index < plan.size(); ++index) {
        fixed.spans[index].first = plan[index].first;
        for (double weight : plan[index].weights) {
            const double value = weight * (1 << fixed.precision);
            fixed.spans[index].weights.push_back(static_cast<std::int16_t>(
                value < 0 ? static_cast<int>(-0.5 + value) : static_cast<int>(0.5 + value)));
        }
    }
    return fixed;
}

std::uint8_t apply_fixed(const std::uint8_t* source, std::ptrdiff_t stride,
                         const Span<std::int32_t>& span, unsigned precision) {
    std::int32_t sum = std::int32_t{1} << (precision - 1);
    for (std::size_t j = 0; j < span.weights.size(); ++j)
        sum += static_cast<std::int32_t>(source[static_cast<std::ptrdiff_t>(j) * stride]) *
               span.weights[j];
    return static_cast<std::uint8_t>(std::clamp(sum >> precision, 0, 255));
}

void check_size(ImageSize size, const char* what) {
    if (size.height <= 0 || size.width <= 0)
        throw std::invalid_argument(std::string("oneformer ") + what + " must be non-empty");
}

} // namespace

ImageSize processed_size(ImageSize source, std::int32_t shortest, std::int32_t longest) {
    check_size(source, "source image");
    if (shortest <= 0 || longest <= shortest)
        throw std::invalid_argument("oneformer resize edges must satisfy 0 < shortest < longest");
    const double height = source.height, width = source.width;
    const double small = std::min(height, width), large = std::max(height, width);
    std::int64_t size = shortest;
    double raw_size = 0.0;
    bool capped = false;
    if (large / small * static_cast<double>(shortest) > static_cast<double>(longest)) {
        raw_size = static_cast<double>(longest) * small / large;
        size = static_cast<std::int64_t>(std::nearbyint(raw_size)); // Python round(): half to even.
        capped = true;
    }
    ImageSize target;
    if ((source.height <= source.width && source.height == size) ||
        (source.width <= source.height && source.width == size)) {
        target = source;
    } else if (source.width < source.height) {
        target.width = static_cast<std::int32_t>(size);
        target.height = static_cast<std::int32_t>(
            capped ? raw_size * height / width : static_cast<double>(size * source.height) / width);
    } else {
        target.height = static_cast<std::int32_t>(size);
        target.width = static_cast<std::int32_t>(
            capped ? raw_size * width / height : static_cast<double>(size * source.width) / height);
    }
    check_size(target, "resized image (extreme aspect ratio)");
    return target;
}

std::vector<std::uint8_t> resize_uint8_antialias(const std::uint8_t* pixels, ImageSize source,
                                                 ImageSize target, std::int32_t channels) {
    check_size(source, "source image");
    check_size(target, "target image");
    if (pixels == nullptr || channels <= 0)
        throw std::invalid_argument("oneformer resize needs interleaved pixels");
    const auto c = static_cast<std::size_t>(channels);
    std::vector<std::uint8_t> current(pixels, pixels + static_cast<std::size_t>(source.height) *
                                                           static_cast<std::size_t>(source.width) *
                                                           c);
    ImageSize size = source;
    if (target.width != size.width) {
        const auto plan = fixed_plan(size.width, target.width);
        std::vector<std::uint8_t> next(static_cast<std::size_t>(size.height) * target.width * c);
        for (std::int32_t y = 0; y < size.height; ++y) {
            const auto* row = current.data() + static_cast<std::size_t>(y) * size.width * c;
            auto* out = next.data() + static_cast<std::size_t>(y) * target.width * c;
            for (std::int32_t x = 0; x < target.width; ++x) {
                const auto& span = plan.spans[static_cast<std::size_t>(x)];
                for (std::size_t k = 0; k < c; ++k)
                    out[static_cast<std::size_t>(x) * c + k] =
                        apply_fixed(row + static_cast<std::size_t>(span.first) * c + k,
                                    static_cast<std::ptrdiff_t>(c), span, plan.precision);
            }
        }
        current.swap(next);
        size.width = target.width;
    }
    if (target.height != size.height) {
        const auto plan = fixed_plan(size.height, target.height);
        const auto stride = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(size.width) * c);
        std::vector<std::uint8_t> next(static_cast<std::size_t>(target.height) * size.width * c);
        for (std::int32_t y = 0; y < target.height; ++y) {
            const auto& span = plan.spans[static_cast<std::size_t>(y)];
            const auto* column =
                current.data() + static_cast<std::size_t>(span.first) * size.width * c;
            auto* out = next.data() + static_cast<std::size_t>(y) * size.width * c;
            for (std::size_t offset = 0; offset < static_cast<std::size_t>(size.width) * c;
                 ++offset)
                out[offset] = apply_fixed(column + offset, stride, span, plan.precision);
        }
        current.swap(next);
    }
    return current;
}

std::vector<float> normalize_planar(const std::vector<std::uint8_t>& pixels, ImageSize size,
                                    const Normalization& normalization) {
    check_size(size, "normalized image");
    const auto plane = static_cast<std::size_t>(size.height) * static_cast<std::size_t>(size.width);
    if (pixels.size() != plane * 3)
        throw std::invalid_argument("oneformer normalization needs RGB pixels");
    // Fused like the fast processor: mean and std scaled by 1 / rescale_factor in float32.
    const auto factor = static_cast<float>(1.0 / normalization.rescale_factor);
    std::vector<float> planar(plane * 3);
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const float mean = normalization.mean[channel] * factor;
        const float deviation = normalization.std[channel] * factor;
        float* out = planar.data() + channel * plane;
        for (std::size_t index = 0; index < plane; ++index)
            out[index] = (static_cast<float>(pixels[index * 3 + channel]) - mean) / deviation;
    }
    return planar;
}

std::vector<std::int32_t> resize_argmax(const float* scores, std::int32_t classes, ImageSize source,
                                        ImageSize target) {
    check_size(source, "score map");
    check_size(target, "label map");
    if (scores == nullptr || classes <= 0)
        throw std::invalid_argument("oneformer argmax needs score maps");
    const auto k = static_cast<std::size_t>(classes);

    // Horizontal pass first (ATen's order), skipped when the width is unchanged.
    std::vector<float> horizontal;
    const float* rows = scores;
    if (target.width != source.width) {
        float max_weight = 0.0F;
        const auto plan = antialias_plan<float>(source.width, target.width, max_weight);
        horizontal.resize(k * source.height * target.width);
        for (std::size_t plane = 0; plane < k * source.height; ++plane) {
            const float* row = scores + plane * source.width;
            float* out = horizontal.data() + plane * target.width;
            for (std::int32_t x = 0; x < target.width; ++x) {
                const auto& span = plan[static_cast<std::size_t>(x)];
                const float* first = row + span.first;
                float value = first[0] * span.weights[0];
                for (std::size_t j = 1; j < span.weights.size(); ++j)
                    value += first[j] * span.weights[j];
                out[x] = value;
            }
        }
        rows = horizontal.data();
    }
    const auto row_plane = static_cast<std::size_t>(source.height) * target.width;

    std::vector<Span<float>> vertical;
    if (target.height != source.height) {
        float max_weight = 0.0F;
        vertical = antialias_plan<float>(source.height, target.height, max_weight);
    }
    std::vector<std::int32_t> labels(static_cast<std::size_t>(target.height) * target.width);
    std::vector<float> best(static_cast<std::size_t>(target.width));
    for (std::int32_t y = 0; y < target.height; ++y) {
        auto* out = labels.data() + static_cast<std::size_t>(y) * target.width;
        for (std::size_t c = 0; c < k; ++c) {
            const float* plane = rows + c * row_plane;
            for (std::int32_t x = 0; x < target.width; ++x) {
                float value;
                if (vertical.empty()) {
                    value = plane[static_cast<std::size_t>(y) * target.width + x];
                } else {
                    const auto& span = vertical[static_cast<std::size_t>(y)];
                    const float* first =
                        plane + static_cast<std::size_t>(span.first) * target.width + x;
                    value = first[0] * span.weights[0];
                    for (std::size_t j = 1; j < span.weights.size(); ++j)
                        value +=
                            first[j * static_cast<std::size_t>(target.width)] * span.weights[j];
                }
                if (c == 0 || value > best[static_cast<std::size_t>(x)]) {
                    best[static_cast<std::size_t>(x)] = value;
                    out[x] = static_cast<std::int32_t>(c);
                }
            }
        }
    }
    return labels;
}

} // namespace trtmc::oneformer
