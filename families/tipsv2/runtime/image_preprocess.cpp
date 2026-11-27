/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/tipsv2/runtime/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace trtmc::tipsv2 {
namespace {

// Pillow's bilinear resize, reproduced coefficient for coefficient. The reference resizes the
// 8-bit PIL image (`resample=2`) and only then divides by 255, so the whole filter runs in
// Pillow's fixed-point integer pipeline over 8-bit samples; a float bilinear lands a few
// least-significant bits away on every pixel, which is enough to move the last digits of a
// cosine. Pillow also scales the filter support by the downscaling ratio, which is what makes its
// resize antialiased - a plain two-tap interpolation of a large photograph is a visibly different
// image, not a rounding difference.
constexpr std::int32_t kPillowPrecisionBits = 22;
constexpr std::int64_t kPillowScale = std::int64_t{1} << kPillowPrecisionBits;
constexpr std::int64_t kPillowRounding = std::int64_t{1} << (kPillowPrecisionBits - 1);
constexpr double kBilinearSupport = 1.0;
// A generous ceiling on the samples either image may hold - eight gigapixels of RGB, far past
// any photograph and far below where the index arithmetic below could wrap.
constexpr std::uint64_t kMaximumSamples = std::uint64_t{1} << 33U;

struct PillowSpan {
    std::int32_t first{0};
    std::vector<std::int32_t> weights;
};

double pillow_bilinear(double value) {
    value = std::abs(value);
    return value < 1.0 ? 1.0 - value : 0.0;
}

std::vector<PillowSpan> make_pillow_plan(std::int32_t input_size, std::int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = kBilinearSupport * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<PillowSpan> plan(static_cast<std::size_t>(output_size));
    for (std::int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        // Clip in double and only then narrow: `center + support + 0.5` for an input size near
        // INT32_MAX is past what an int32 holds, and the conversion of such a value is undefined
        // rather than merely wrong.
        const auto clip = [input_size](double value) {
            return static_cast<std::int32_t>(
                std::min(std::max(value, 0.0), static_cast<double>(input_size)));
        };
        const std::int32_t first = clip(center - support + 0.5);
        const std::int32_t end = clip(center + support + 0.5);
        if (end <= first) {
            throw std::runtime_error("tipsv2 Pillow resize produced an empty support");
        }

        auto& span = plan[static_cast<std::size_t>(output_index)];
        span.first = first;
        std::vector<double> floating(static_cast<std::size_t>(end - first));
        double total = 0.0;
        for (std::int32_t index = first; index < end; ++index) {
            const double weight =
                pillow_bilinear((static_cast<double>(index) - center + 0.5) * inverse_filter_scale);
            floating[static_cast<std::size_t>(index - first)] = weight;
            total += weight;
        }
        if (!std::isfinite(total) || total == 0.0) {
            throw std::runtime_error("tipsv2 Pillow resize has invalid coefficients");
        }
        span.weights.reserve(floating.size());
        for (const double weight : floating) {
            const double scaled = weight / total * static_cast<double>(kPillowScale);
            if (!std::isfinite(scaled) ||
                scaled < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
                scaled > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
                throw std::runtime_error("tipsv2 Pillow resize coefficient overflowed");
            }
            span.weights.push_back(
                static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
        }
    }
    return plan;
}

std::uint8_t apply_pillow_span(const std::uint8_t* source, std::size_t stride,
                               const PillowSpan& span) {
    std::int64_t sum = kPillowRounding;
    for (std::size_t index = 0; index < span.weights.size(); ++index) {
        // `(first + index) * stride` in int32 wraps on a large image; the offset is computed in
        // size_t, which is what indexes the buffer.
        const std::size_t offset = (static_cast<std::size_t>(span.first) + index) * stride;
        sum += static_cast<std::int64_t>(source[offset]) * span.weights[index];
    }
    if (sum <= 0) {
        return 0;
    }
    return static_cast<std::uint8_t>(std::min<std::int64_t>(sum >> kPillowPrecisionBits, 255));
}

// Resample every row to `output_w`, keeping the height.
std::vector<std::uint8_t> horizontal_pass(const std::vector<std::uint8_t>& input,
                                          std::int32_t height, std::int32_t input_w,
                                          std::int32_t output_w) {
    const auto plan = make_pillow_plan(input_w, output_w);
    std::vector<std::uint8_t> output(static_cast<std::size_t>(height) * output_w * 3U);
    for (std::int32_t y = 0; y < height; ++y) {
        for (std::int32_t x = 0; x < output_w; ++x) {
            const auto& span = plan[static_cast<std::size_t>(x)];
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto source = static_cast<std::size_t>(y) * input_w * 3U + channel;
                const auto target = (static_cast<std::size_t>(y) * output_w + x) * 3U + channel;
                output[target] = apply_pillow_span(input.data() + source, 3, span);
            }
        }
    }
    return output;
}

// Resample every column to `output_h`, keeping the width.
std::vector<std::uint8_t> vertical_pass(const std::vector<std::uint8_t>& input,
                                        std::int32_t input_h, std::int32_t width,
                                        std::int32_t output_h) {
    const auto plan = make_pillow_plan(input_h, output_h);
    std::vector<std::uint8_t> output(static_cast<std::size_t>(output_h) * width * 3U);
    for (std::int32_t y = 0; y < output_h; ++y) {
        const auto& span = plan[static_cast<std::size_t>(y)];
        for (std::int32_t x = 0; x < width; ++x) {
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto source = static_cast<std::size_t>(x) * 3U + channel;
                const auto target = (static_cast<std::size_t>(y) * width + x) * 3U + channel;
                output[target] = apply_pillow_span(input.data() + source,
                                                   static_cast<std::size_t>(width) * 3U, span);
            }
        }
    }
    return output;
}

// Which axis Pillow resamples first. The two passes are not interchangeable: the intermediate is
// 8-bit, so the order moves a few samples by one level. Transcribed from `ImagingResampleInner`
// in Pillow's `Resample.c`, where it is an optimisation ("if height is being scaled down more
// than twice the amount the width is, run the vertical pass first, to make the horizontal pass
// faster") that is also observable in the output. Pillow 12 and earlier always ran the
// horizontal pass first; `families/tipsv2/requirements.txt` pins the version this reproduces.
bool horizontal_first(std::int32_t input_h, std::int32_t input_w, std::int32_t output_h,
                      std::int32_t output_w) {
    const std::int64_t height_shrink = std::int64_t{input_h} - output_h;
    const std::int64_t width_shrink = std::int64_t{input_w} - output_w;
    return !(height_shrink > 0 && height_shrink > width_shrink * 2);
}

std::vector<std::uint8_t> resize_pillow_bilinear(const float* source, std::int32_t input_h,
                                                 std::int32_t input_w, std::int32_t output_h,
                                                 std::int32_t output_w) {
    std::vector<std::uint8_t> input(static_cast<std::size_t>(input_h) * input_w * 3U);
    std::transform(source, source + input.size(), input.begin(), [](float value) {
        if (!std::isfinite(value)) {
            // A NaN or infinity would make `lround` undefined and then index arithmetic on it is
            // meaningless; reject it at the boundary instead.
            throw std::invalid_argument("tipsv2 source image has a non-finite sample");
        }
        return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
    });

    // A pass whose size does not change is the identity - its plan is one tap of weight one - so
    // both always run and the order is the only thing that matters.
    if (horizontal_first(input_h, input_w, output_h, output_w)) {
        return vertical_pass(horizontal_pass(input, input_h, input_w, output_w), input_h, output_w,
                             output_h);
    }
    return horizontal_pass(vertical_pass(input, input_h, input_w, output_h), output_h, input_w,
                           output_w);
}

} // namespace

std::vector<float> preprocess_tipsv2_image(const float* image_pixels, std::int32_t image_height,
                                           std::int32_t image_width,
                                           const PreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("tipsv2 source image must be non-empty");
    }
    if (config.input_image_h <= 0 || config.input_image_w <= 0) {
        throw std::invalid_argument("tipsv2 input dimensions must be positive");
    }
    if (!std::isfinite(config.rescale_factor) || config.rescale_factor <= 0.0F) {
        throw std::invalid_argument("tipsv2 rescale factor must be positive and finite");
    }
    // Bound the work before doing it, and keep every offset below in range: the two passes index
    // `height * width * 3` samples, and a plan's `first + index` has to stay inside that.
    const auto samples =
        static_cast<std::uint64_t>(image_height) * static_cast<std::uint64_t>(image_width) * 3U;
    const auto resized = static_cast<std::uint64_t>(config.input_image_h) *
                         static_cast<std::uint64_t>(config.input_image_w) * 3U;
    if (samples > kMaximumSamples || resized > kMaximumSamples) {
        throw std::invalid_argument("tipsv2 image is larger than this family resamples");
    }

    // Straight to the target in both axes: a 640x480 source becomes 448x448 and every pixel of
    // the original is represented, squashed. Preserving the aspect ratio here - the habit a CLIP
    // seam brings with it - would feed the tower a different image than the reference saw.
    const auto squashed = resize_pillow_bilinear(image_pixels, image_height, image_width,
                                                 config.input_image_h, config.input_image_w);

    const auto plane = static_cast<std::size_t>(config.input_image_h) * config.input_image_w;
    std::vector<float> pixel_values(3U * plane);
    for (std::int32_t y = 0; y < config.input_image_h; ++y) {
        for (std::int32_t x = 0; x < config.input_image_w; ++x) {
            const std::size_t source =
                (static_cast<std::size_t>(y) * config.input_image_w + x) * 3U;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                pixel_values[channel * plane + static_cast<std::size_t>(y) * config.input_image_w +
                             x] =
                    static_cast<float>(squashed[source + channel]) * config.rescale_factor;
            }
        }
    }
    return pixel_values;
}

} // namespace trtmc::tipsv2
