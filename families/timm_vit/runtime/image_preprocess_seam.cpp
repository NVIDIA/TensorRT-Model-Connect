/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timm_vit/runtime/image_preprocess_seam.h"

#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

namespace {

constexpr std::int32_t kPillowPrecisionBits = 22;
constexpr std::int64_t kPillowScale = std::int64_t{1} << kPillowPrecisionBits;
constexpr std::int64_t kPillowRounding = std::int64_t{1} << (kPillowPrecisionBits - 1);

// One axis of the Pillow resize. Output index i reads count[i] source samples from
// first[i] on, with the fixed-point weights at weights[offset[i]] on.
struct PillowPlan {
    std::vector<std::int32_t> first;
    std::vector<std::int32_t> count;
    std::vector<std::size_t> offset;
    std::vector<std::int32_t> weights;
};

double pillow_cubic(double value) {
    constexpr double kA = -0.5;
    value = std::abs(value);
    if (value < 1.0)
        return ((kA + 2.0) * value - (kA + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return ((kA * value - 5.0 * kA) * value + 8.0 * kA) * value - 4.0 * kA;
    return 0.0;
}

PillowPlan make_pillow_plan(std::int32_t input_size, std::int32_t output_size) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    PillowPlan plan;
    plan.first.resize(static_cast<std::size_t>(output_size));
    plan.count.resize(static_cast<std::size_t>(output_size));
    plan.offset.resize(static_cast<std::size_t>(output_size));
    std::vector<double> floating;
    for (std::int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const auto first =
            std::max<std::int32_t>(0, static_cast<std::int32_t>(center - support + 0.5));
        const auto end =
            std::min<std::int32_t>(input_size, static_cast<std::int32_t>(center + support + 0.5));
        if (end <= first)
            throw std::runtime_error("timm ViT Pillow resize produced empty support");

        floating.resize(static_cast<std::size_t>(end - first));
        double total = 0.0;
        for (std::int32_t index = first; index < end; ++index) {
            const double weight =
                pillow_cubic((static_cast<double>(index) - center + 0.5) * inverse_filter_scale);
            floating[static_cast<std::size_t>(index - first)] = weight;
            total += weight;
        }
        if (!std::isfinite(total) || total == 0.0)
            throw std::runtime_error("timm ViT Pillow resize has invalid coefficients");
        const auto slot = static_cast<std::size_t>(output_index);
        plan.first[slot] = first;
        plan.count[slot] = end - first;
        plan.offset[slot] = plan.weights.size();
        for (double weight : floating) {
            const double scaled = weight / total * static_cast<double>(kPillowScale);
            if (!std::isfinite(scaled) ||
                scaled < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
                scaled > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
                throw std::runtime_error("timm ViT Pillow resize coefficient overflowed");
            }
            plan.weights.push_back(
                static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
        }
    }
    return plan;
}

// One source row as bytes: std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F) for every
// value but a NaN, which reads as 0. Two loops, so that each vectorizes: the first only
// selects, the second only computes. The clamped value times 255 lies in [0, 255],
// doubling it is exact, and floor((floor(2 * s) + 1) / 2) rounds s to nearest with halves
// up, as lround does there. The second loop holds no floating-point add, so no compiler can
// contract it into another result.
void to_pillow_bytes(const float* source, float* clamped, std::uint8_t* bytes, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        const float positive = source[index] > 0.0F ? source[index] : 0.0F;
        clamped[index] = positive < 1.0F ? positive : 1.0F;
    }
    for (std::size_t index = 0; index < count; ++index) {
        bytes[index] = static_cast<std::uint8_t>(
            (static_cast<std::int32_t>(clamped[index] * 255.0F * 2.0F) + 1) >> 1);
    }
}

std::uint8_t pillow_sum_to_byte(std::int64_t sum) {
    if (sum <= 0)
        return 0;
    const std::int64_t value = sum >> kPillowPrecisionBits;
    return static_cast<std::uint8_t>(value < 255 ? value : 255);
}

stbir_filter resolve_timm_vit_resize_filter(const std::string& interpolation) {
    if (interpolation == "bilinear")
        return STBIR_FILTER_TRIANGLE;
    if (interpolation == "bicubic")
        return STBIR_FILTER_CATMULLROM;
    throw std::invalid_argument("Unsupported timm ViT interpolation: " + interpolation);
}

void validate_timm_vit_preprocess_config(const TimmVitPreprocessConfig& config) {
    if (config.input_image_h <= 0 || config.input_image_w <= 0) {
        throw std::invalid_argument("timm ViT input dimensions must be positive");
    }
    if (config.crop_pct <= 0.0F || config.crop_pct > 1.0F) {
        throw std::invalid_argument("timm ViT crop_pct must be in (0, 1]");
    }
    if (config.image_mean.size() != 3 || config.image_std.size() != 3) {
        throw std::invalid_argument("timm ViT image mean/std must contain three channels");
    }
    for (float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("timm ViT image std must be non-zero");
    }
}

int32_t torchvision_center_crop_offset(int32_t resized, int32_t target) {
    const int32_t difference = resized - target;
    const int32_t half = difference / 2;
    // torchvision uses Python round(), whose .5 ties round to the nearest even integer.
    return (difference % 2 != 0 && half % 2 != 0) ? half + 1 : half;
}

// Pillow's two-pass 8-bit bicubic resize, torchvision's center crop and the
// normalization, computed for the cropped output only. Every output value is the one the
// whole-image passes give: the same weights, the same integer sums (an integer sum does
// not depend on the order of its terms), the same rounding and the same float expression.
std::vector<float> preprocess_pillow_bicubic(const float* source, std::int32_t input_h,
                                             std::int32_t input_w, std::int32_t resized_h,
                                             std::int32_t resized_w,
                                             const TimmVitPreprocessConfig& config) {
    const std::int32_t output_h = config.input_image_h;
    const std::int32_t output_w = config.input_image_w;
    const std::int32_t crop_y = torchvision_center_crop_offset(resized_h, output_h);
    const std::int32_t crop_x = torchvision_center_crop_offset(resized_w, output_w);

    // The source columns the cropped output columns read.
    const auto horizontal_plan = make_pillow_plan(input_w, resized_w);
    std::int32_t column_begin = input_w;
    std::int32_t column_end = 0;
    for (std::int32_t x = 0; x < output_w; ++x) {
        const auto slot = static_cast<std::size_t>(crop_x + x);
        const std::int32_t first = horizontal_plan.first[slot];
        const std::int32_t end = first + horizontal_plan.count[slot];
        column_begin = first < column_begin ? first : column_begin;
        column_end = end > column_end ? end : column_end;
    }
    // The source rows the cropped output rows read.
    const auto vertical_plan = make_pillow_plan(input_h, resized_h);
    std::int32_t row_begin = input_h;
    std::int32_t row_end = 0;
    for (std::int32_t y = 0; y < output_h; ++y) {
        const auto slot = static_cast<std::size_t>(crop_y + y);
        const std::int32_t first = vertical_plan.first[slot];
        const std::int32_t end = first + vertical_plan.count[slot];
        row_begin = first < row_begin ? first : row_begin;
        row_end = end > row_end ? end : row_end;
    }

    // Horizontal pass: one source row to bytes, then one sum per channel of each pixel.
    const auto row_values = static_cast<std::size_t>(output_w) * 3U;
    const auto source_values = static_cast<std::size_t>(column_end - column_begin) * 3U;
    std::vector<float> source_clamped(source_values);
    std::vector<std::uint8_t> source_bytes(source_values);
    std::vector<std::uint8_t> horizontal(static_cast<std::size_t>(row_end - row_begin) *
                                         row_values);
    std::uint8_t* bytes = source_bytes.data();
    for (std::int32_t y = row_begin; y < row_end; ++y) {
        const float* source_row =
            source + (static_cast<std::size_t>(y) * input_w + column_begin) * 3U;
        to_pillow_bytes(source_row, source_clamped.data(), bytes, source_values);

        std::uint8_t* target =
            horizontal.data() + static_cast<std::size_t>(y - row_begin) * row_values;
        for (std::int32_t x = 0; x < output_w; ++x) {
            const auto slot = static_cast<std::size_t>(crop_x + x);
            const std::int32_t* weights =
                horizontal_plan.weights.data() + horizontal_plan.offset[slot];
            const std::uint8_t* taps =
                bytes + static_cast<std::size_t>(horizontal_plan.first[slot] - column_begin) * 3U;
            const std::int32_t count = horizontal_plan.count[slot];
            std::int64_t red = kPillowRounding;
            std::int64_t green = kPillowRounding;
            std::int64_t blue = kPillowRounding;
            for (std::int32_t tap = 0; tap < count; ++tap) {
                const std::int64_t weight = weights[tap];
                red += taps[tap * 3] * weight;
                green += taps[tap * 3 + 1] * weight;
                blue += taps[tap * 3 + 2] * weight;
            }
            target[x * 3] = pillow_sum_to_byte(red);
            target[x * 3 + 1] = pillow_sum_to_byte(green);
            target[x * 3 + 2] = pillow_sum_to_byte(blue);
        }
    }

    // The normalized value of each byte, by the per-pixel expression: a byte to [0, 1],
    // minus the mean, over the std.
    std::array<std::array<float, 256>, 3> normalized{};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        for (std::size_t level = 0; level < 256; ++level) {
            normalized[channel][level] =
                (static_cast<float>(level) / 255.0F - config.image_mean[channel]) /
                config.image_std[channel];
        }
    }

    // Vertical pass: a whole-row sweep per tap into one row of sums, then the planar,
    // normalized output row.
    const auto output_plane = static_cast<std::size_t>(output_h) * output_w;
    std::vector<float> pixel_values(3U * output_plane);
    std::vector<std::int64_t> sums(row_values);
    for (std::int32_t y = 0; y < output_h; ++y) {
        const auto slot = static_cast<std::size_t>(crop_y + y);
        const std::int32_t* weights = vertical_plan.weights.data() + vertical_plan.offset[slot];
        const std::int32_t count = vertical_plan.count[slot];
        std::int64_t* row_sums = sums.data();
        for (std::size_t index = 0; index < row_values; ++index)
            row_sums[index] = kPillowRounding;
        for (std::int32_t tap = 0; tap < count; ++tap) {
            const std::uint8_t* row =
                horizontal.data() +
                static_cast<std::size_t>(vertical_plan.first[slot] + tap - row_begin) * row_values;
            const std::int64_t weight = weights[tap];
            for (std::size_t index = 0; index < row_values; ++index)
                row_sums[index] += row[index] * weight;
        }
        for (std::size_t channel = 0; channel < 3; ++channel) {
            float* output = pixel_values.data() + channel * output_plane +
                            static_cast<std::size_t>(y) * output_w;
            const float* table = normalized[channel].data();
            for (std::int32_t x = 0; x < output_w; ++x)
                output[x] =
                    table[pillow_sum_to_byte(row_sums[static_cast<std::size_t>(x) * 3U + channel])];
        }
    }
    return pixel_values;
}

} // namespace

TimmVitResizeShape compute_timm_vit_resize_shape(int32_t image_height, int32_t image_width,
                                                 const TimmVitPreprocessConfig& config) {
    if (image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("timm ViT source dimensions must be positive");
    }
    validate_timm_vit_preprocess_config(config);

    if (config.input_image_h == config.input_image_w) {
        // timm's center-crop eval transform passes floor(input_size / crop_pct) as a scalar
        // torchvision Resize size. A scalar fixes the shorter edge and floors the aspect-ratio
        // calculation for the longer edge.
        const int32_t resized_short = static_cast<int32_t>(
            std::floor(static_cast<float>(config.input_image_h) / config.crop_pct));
        if (image_height <= image_width) {
            return {resized_short, static_cast<int32_t>(static_cast<int64_t>(resized_short) *
                                                        image_width / image_height)};
        }
        return {
            static_cast<int32_t>(static_cast<int64_t>(resized_short) * image_height / image_width),
            resized_short};
    }

    const float required_scale =
        std::max(static_cast<float>(config.input_image_h) / static_cast<float>(image_height),
                 static_cast<float>(config.input_image_w) / static_cast<float>(image_width));
    const float resize_scale = required_scale / config.crop_pct;
    return {
        std::max(config.input_image_h,
                 static_cast<int32_t>(std::floor(static_cast<float>(image_height) * resize_scale))),
        std::max(config.input_image_w,
                 static_cast<int32_t>(std::floor(static_cast<float>(image_width) * resize_scale))),
    };
}

std::vector<float> preprocess_timm_vit_image(const float* image_pixels, int32_t image_height,
                                             int32_t image_width,
                                             const TimmVitPreprocessConfig& config) {
    if (image_pixels == nullptr || image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("timm ViT source image must be non-empty");
    }
    validate_timm_vit_preprocess_config(config);

    const auto resize_shape = compute_timm_vit_resize_shape(image_height, image_width, config);
    const int32_t resized_h = resize_shape.height;
    const int32_t resized_w = resize_shape.width;

    std::vector<float> resized;
    if (config.interpolation == "bicubic") {
        return preprocess_pillow_bicubic(image_pixels, image_height, image_width, resized_h,
                                         resized_w, config);
    } else {
        resized.resize(static_cast<std::size_t>(resized_h) * resized_w * 3U);
        if (stbir_resize(image_pixels, image_width, image_height,
                         image_width * 3 * static_cast<int32_t>(sizeof(float)), resized.data(),
                         resized_w, resized_h, resized_w * 3 * static_cast<int32_t>(sizeof(float)),
                         STBIR_RGB, STBIR_TYPE_FLOAT, STBIR_EDGE_CLAMP,
                         resolve_timm_vit_resize_filter(config.interpolation)) == nullptr) {
            throw std::runtime_error("Failed to resize timm ViT input image");
        }
    }

    const int32_t crop_y = torchvision_center_crop_offset(resized_h, config.input_image_h);
    const int32_t crop_x = torchvision_center_crop_offset(resized_w, config.input_image_w);
    const auto output_plane = static_cast<std::size_t>(config.input_image_h) * config.input_image_w;
    std::vector<float> pixel_values(3U * output_plane);
    for (int32_t y = 0; y < config.input_image_h; ++y) {
        for (int32_t x = 0; x < config.input_image_w; ++x) {
            const auto src_idx =
                static_cast<std::size_t>((((crop_y + y) * resized_w + crop_x + x) * 3));
            for (int32_t c = 0; c < 3; ++c) {
                const auto channel = static_cast<std::size_t>(c);
                pixel_values[channel * output_plane +
                             static_cast<std::size_t>(y) * config.input_image_w + x] =
                    (resized[src_idx + channel] - config.image_mean[channel]) /
                    config.image_std[channel];
            }
        }
    }
    return pixel_values;
}

} // namespace trtmc
