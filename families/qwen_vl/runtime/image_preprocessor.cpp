/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_vl/runtime/image_preprocessor.h"

#include "image_transform_helper.h"
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace trtmc {

// ---------------------------------------------------------------------------
// Interpolation filter resolution
// ---------------------------------------------------------------------------

static stbir_filter resolve_stbir_filter(const std::string& interpolation) {
    if (interpolation == "bilinear")
        return STBIR_FILTER_TRIANGLE;
    if (interpolation == "nearest")
        return STBIR_FILTER_POINT_SAMPLE;
    // "bicubic" or anything else -> Catmull-Rom. Qwen's native dynamic path
    // uses the PyTorch-compatible antialiased implementation below instead.
    return STBIR_FILTER_CATMULLROM;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

struct LoadedImage {
    std::vector<float> img_chw; // [C, H, W] normalized
    int target_height{0};
    int target_width{0};
    int channels{0};
    bool ok{false};
};

static int32_t preprocess_worker_count() {
    static const int32_t workers = []() -> int32_t {
        constexpr unsigned worker_cap = 8;
        const unsigned hardware = std::thread::hardware_concurrency();
        return static_cast<int32_t>(hardware == 0 ? worker_cap : std::min(hardware, worker_cap));
    }();
    return workers;
}

template <typename Function>
static void parallel_for_ranges(int32_t count, int32_t min_grain, Function&& function) {
    if (count <= 0)
        return;
    const int32_t by_grain = (count + min_grain - 1) / min_grain;
    const int32_t workers = std::min({count, by_grain, preprocess_worker_count()});
    if (workers <= 1) {
        function(0, count);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers - 1));
    for (int32_t worker = 0; worker < workers - 1; ++worker) {
        const int32_t begin = count * worker / workers;
        const int32_t end = count * (worker + 1) / workers;
        threads.emplace_back([&, begin, end] { function(begin, end); });
    }
    function(count * (workers - 1) / workers, count);
    for (auto& thread : threads)
        thread.join();
}

static int target_height(const QwenVlPreprocessConfig& config) {
    return config.fixed_image_height > 0 ? config.fixed_image_height : config.fixed_image_size;
}

static int target_width(const QwenVlPreprocessConfig& config) {
    return config.fixed_image_width > 0 ? config.fixed_image_width : config.fixed_image_size;
}

std::array<int32_t, 2> qwen_vl_smart_resize(int32_t image_height, int32_t image_width,
                                            int32_t factor, int32_t min_pixels,
                                            int32_t max_pixels) {
    if (image_height <= 0 || image_width <= 0 || factor <= 0 || min_pixels <= 0 ||
        max_pixels < min_pixels) {
        throw std::invalid_argument("invalid Qwen-VL smart-resize configuration");
    }
    const double long_side = static_cast<double>(std::max(image_height, image_width));
    const double short_side = static_cast<double>(std::min(image_height, image_width));
    if (long_side / short_side > 200.0)
        throw std::invalid_argument("Qwen-VL image aspect ratio exceeds 200");

    const auto aligned_round = [factor](double extent) {
        return std::max(factor, static_cast<int32_t>(std::nearbyint(extent / factor)) * factor);
    };
    int32_t height = aligned_round(image_height);
    int32_t width = aligned_round(image_width);
    const int64_t aligned_pixels = static_cast<int64_t>(height) * width;
    const double source_pixels = static_cast<double>(image_height) * image_width;
    if (aligned_pixels > max_pixels) {
        const double beta = std::sqrt(source_pixels / max_pixels);
        height = std::max(factor,
                          static_cast<int32_t>(std::floor(image_height / beta / factor)) * factor);
        width = std::max(factor,
                         static_cast<int32_t>(std::floor(image_width / beta / factor)) * factor);
    } else if (aligned_pixels < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) / source_pixels);
        height = std::max(factor,
                          static_cast<int32_t>(std::ceil(image_height * beta / factor)) * factor);
        width =
            std::max(factor, static_cast<int32_t>(std::ceil(image_width * beta / factor)) * factor);
    }
    return {height, width};
}

// Resize raw uint8 RGB pixels to the fixed vision profile dimensions.
static std::vector<unsigned char> resize_raw(const unsigned char* raw, int width, int height,
                                             int target_width, int target_height,
                                             stbir_filter filter) {
    std::vector<unsigned char> resized(static_cast<std::size_t>(target_width) * target_height * 3);

    void* result =
        stbir_resize(raw, width, height, width * 3, resized.data(), target_width, target_height,
                     target_width * 3, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, filter);

    if (result == nullptr) {
        return {};
    }
    return resized;
}

// Qwen2-VL's fast Hugging Face processor resizes uint8 tensors with
// torchvision bicubic interpolation and antialiasing enabled. In particular,
// downsampling widens the cubic support; a plain Catmull-Rom resize does not.
// The fixed-point coefficient conversion and per-axis uint8 rounding below
// mirror PyTorch's CPU uint8 path so preprocessing does not drift before the
// vision transformer.
struct AntialiasWeights {
    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int16_t> values;
    int32_t stride{0};
    uint32_t precision{0};
};

static double keys_cubic(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

static int32_t aligned_coefficient_stride(int32_t kernel_size) {
    int32_t stride = kernel_size;
    while (stride % static_cast<int32_t>(sizeof(int32_t)) != 0)
        ++stride;
    return stride;
}

static double build_floating_antialias_weights(int32_t input_size, int32_t output_size,
                                               double scale, double support, int32_t kernel_size,
                                               AntialiasWeights& result,
                                               std::vector<double>& floating) {
    double maximum_weight = 0.0;
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = scale * (output_index + 0.5);
        const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
        const int32_t start = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        const int32_t size =
            std::clamp(std::min(static_cast<int32_t>(center + support + 0.5), input_size) - start,
                       0, kernel_size);
        result.starts[static_cast<std::size_t>(output_index)] = start;
        result.sizes[static_cast<std::size_t>(output_index)] = size;

        double total = 0.0;
        double* row = floating.data() + static_cast<std::size_t>(output_index) * kernel_size;
        for (int32_t index = 0; index < size; ++index) {
            row[index] = keys_cubic((index + start - center + 0.5) * inverse_scale);
            total += row[index];
        }
        if (total != 0.0) {
            for (int32_t index = 0; index < size; ++index) {
                row[index] /= total;
                maximum_weight = std::max(maximum_weight, row[index]);
            }
        }
    }
    return maximum_weight;
}

static void quantize_antialias_weights(const std::vector<double>& floating, int32_t output_size,
                                       int32_t kernel_size, double maximum_weight,
                                       AntialiasWeights& result) {
    // Select the greatest fixed-point precision whose largest coefficient fits
    // in int16, exactly as PyTorch's uint8 antialias implementation does.
    for (; result.precision < 22; ++result.precision) {
        const int32_t next = static_cast<int32_t>(
            0.5 + maximum_weight * static_cast<double>(uint32_t{1} << (result.precision + 1)));
        if (next >= (1 << 15))
            break;
    }

    const double multiplier = static_cast<double>(uint32_t{1} << result.precision);
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double* source =
            floating.data() + static_cast<std::size_t>(output_index) * kernel_size;
        int16_t* destination =
            result.values.data() + static_cast<std::size_t>(output_index) * result.stride;
        for (int32_t index = 0; index < kernel_size; ++index) {
            const double value = source[index] * multiplier;
            destination[index] =
                static_cast<int16_t>(value < 0.0 ? static_cast<int32_t>(value - 0.5)
                                                 : static_cast<int32_t>(value + 0.5));
        }
    }
}

static AntialiasWeights make_bicubic_antialias_weights(int32_t input_size, int32_t output_size) {
    constexpr int32_t interp_size = 4;
    const double scale = static_cast<double>(input_size) / output_size;
    const double support = scale >= 1.0 ? (interp_size * 0.5) * scale : interp_size * 0.5;
    const int32_t kernel_size = static_cast<int32_t>(std::ceil(support)) * 2 + 1;

    // PyTorch pads each int16 coefficient row to a 32-bit-aligned size in its
    // optimized uint8 path. The padding does not participate in convolution.
    AntialiasWeights result;
    result.stride = aligned_coefficient_stride(kernel_size);
    result.starts.resize(static_cast<std::size_t>(output_size));
    result.sizes.resize(static_cast<std::size_t>(output_size));
    result.values.assign(static_cast<std::size_t>(output_size) * result.stride, 0);

    std::vector<double> floating(static_cast<std::size_t>(output_size) * kernel_size, 0.0);
    const double maximum_weight = build_floating_antialias_weights(
        input_size, output_size, scale, support, kernel_size, result, floating);
    quantize_antialias_weights(floating, output_size, kernel_size, maximum_weight, result);
    return result;
}

static uint8_t fixed_point_pixel(const uint8_t* source, int32_t source_stride,
                                 const int16_t* weights, int32_t count, uint32_t precision) {
    int32_t value = int32_t{1} << (precision - 1);
    for (int32_t index = 0; index < count; ++index)
        value += static_cast<int32_t>(source[static_cast<std::size_t>(index) * source_stride]) *
                 weights[index];
    return static_cast<uint8_t>(std::clamp(value >> precision, 0, 255));
}

static bool valid_resize_dimensions(const unsigned char* raw, int32_t width, int32_t height,
                                    int32_t target_width, int32_t target_height) {
    return raw != nullptr && width > 0 && height > 0 && target_width > 0 && target_height > 0;
}

static std::vector<unsigned char> resize_bicubic_horizontal(const unsigned char* raw, int32_t width,
                                                            int32_t height, int32_t target_width) {
    const auto weights = make_bicubic_antialias_weights(width, target_width);
    std::vector<unsigned char> resized(static_cast<std::size_t>(height) * target_width * 3);
    parallel_for_ranges(height, 16, [&](int32_t begin_y, int32_t end_y) {
        for (int32_t y = begin_y; y < end_y; ++y) {
            for (int32_t x = 0; x < target_width; ++x) {
                const int32_t start = weights.starts[static_cast<std::size_t>(x)];
                const int32_t count = weights.sizes[static_cast<std::size_t>(x)];
                const int16_t* coefficients =
                    weights.values.data() + static_cast<std::size_t>(x) * weights.stride;
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const auto source_offset =
                        (static_cast<std::size_t>(y) * width + start) * 3 + channel;
                    const auto destination_offset =
                        (static_cast<std::size_t>(y) * target_width + x) * 3 + channel;
                    resized[destination_offset] = fixed_point_pixel(
                        raw + source_offset, 3, coefficients, count, weights.precision);
                }
            }
        }
    });
    return resized;
}

static std::vector<unsigned char> resize_bicubic_vertical(const unsigned char* raw, int32_t width,
                                                          int32_t height, int32_t target_height) {
    const auto weights = make_bicubic_antialias_weights(height, target_height);
    std::vector<unsigned char> resized(static_cast<std::size_t>(target_height) * width * 3);
    const int32_t row_stride = width * 3;
    parallel_for_ranges(target_height, 16, [&](int32_t begin_y, int32_t end_y) {
        for (int32_t y = begin_y; y < end_y; ++y) {
            const int32_t start = weights.starts[static_cast<std::size_t>(y)];
            const int32_t count = weights.sizes[static_cast<std::size_t>(y)];
            const int16_t* coefficients =
                weights.values.data() + static_cast<std::size_t>(y) * weights.stride;
            for (int32_t x = 0; x < width; ++x) {
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const auto source_offset =
                        (static_cast<std::size_t>(start) * width + x) * 3 + channel;
                    const auto destination_offset =
                        (static_cast<std::size_t>(y) * width + x) * 3 + channel;
                    resized[destination_offset] = fixed_point_pixel(
                        raw + source_offset, row_stride, coefficients, count, weights.precision);
                }
            }
        }
    });
    return resized;
}

std::vector<unsigned char> qwen_vl_resize_bicubic_antialias_u8(const unsigned char* raw,
                                                               int32_t width, int32_t height,
                                                               int32_t target_width,
                                                               int32_t target_height) {
    if (!valid_resize_dimensions(raw, width, height, target_width, target_height))
        return {};
    if (width == target_width && height == target_height) {
        return {raw, raw + static_cast<std::size_t>(width) * height * 3};
    }

    std::vector<unsigned char> horizontal;
    const unsigned char* vertical_source = raw;
    if (width != target_width) {
        horizontal = resize_bicubic_horizontal(raw, width, height, target_width);
        vertical_source = horizontal.data();
    }

    if (height == target_height)
        return horizontal;

    return resize_bicubic_vertical(vertical_source, target_width, height, target_height);
}

// Convert resized uint8 HWC buffer to float32 CHW, normalizing per channel.
static bool normalize_to_chw(const std::vector<unsigned char>& resized, int target_width,
                             int target_height, const QwenVlPreprocessConfig& config,
                             std::vector<float>& out_chw) {
    if (target_width <= 0 || target_height <= 0 || config.in_channels <= 0 ||
        config.in_channels > 3) {
        return false;
    }
    const std::size_t pixel_count = static_cast<std::size_t>(target_width) * target_height;
    const std::size_t required_size = pixel_count * config.in_channels;
    if (resized.size() < required_size)
        return false;

    out_chw.resize(required_size);
    parallel_for_ranges(
        config.in_channels * target_height, 16, [&](int32_t begin_row, int32_t end_row) {
            for (int32_t row = begin_row; row < end_row; ++row) {
                const int32_t channel = row / target_height;
                const int32_t y = row % target_height;
                const float mean = config.image_mean[channel];
                const float standard_deviation = config.image_std[channel];
                const float inverse_standard_deviation =
                    standard_deviation > 1.0e-8F ? 1.0F / standard_deviation : 1.0F;
                const std::size_t source_row =
                    static_cast<std::size_t>(y) * target_width * config.in_channels + channel;
                const std::size_t destination_row =
                    static_cast<std::size_t>(channel) * pixel_count +
                    static_cast<std::size_t>(y) * target_width;
                for (int32_t x = 0; x < target_width; ++x) {
                    const float pixel =
                        static_cast<float>(resized[source_row + static_cast<std::size_t>(x) *
                                                                    config.in_channels]) /
                        255.0F;
                    out_chw[destination_row + x] = (pixel - mean) * inverse_standard_deviation;
                }
            }
        });
    return true;
}

// ---------------------------------------------------------------------------
// Load strategies
// ---------------------------------------------------------------------------

static LoadedImage load_resize_normalize(const runtime::adapters::io::DecodedImage& image,
                                         const QwenVlPreprocessConfig& config) {
    LoadedImage loaded;

    if (image.empty()) {
        std::cerr << "[trtmc] Failed to preprocess image: decoded image missing" << std::endl;
        return loaded;
    }

    const int dst_h = target_height(config);
    const int dst_w = target_width(config);

    auto resized = resize_raw(image.pixels.data(), image.width, image.height, dst_w, dst_h,
                              resolve_stbir_filter(config.interpolation));

    if (resized.empty()) {
        std::cerr << "[trtmc] Failed to resize image" << std::endl;
        return loaded;
    }

    // 3. Normalize to [C, H, W]
    if (!normalize_to_chw(resized, dst_w, dst_h, config, loaded.img_chw)) {
        std::cerr << "[trtmc] Failed to normalize image" << std::endl;
        return loaded;
    }
    loaded.target_height = dst_h;
    loaded.target_width = dst_w;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

static LoadedImage load_smart_resize_normalize(const runtime::adapters::io::DecodedImage& image,
                                               const QwenVlPreprocessConfig& config) {
    LoadedImage loaded;
    if (image.empty()) {
        std::cerr << "[trtmc] Failed to preprocess smart-resized image: decoded image missing"
                  << std::endl;
        return loaded;
    }
    std::array<int32_t, 2> target{};
    try {
        target =
            qwen_vl_smart_resize(image.height, image.width, config.patch_size * config.merge_size,
                                 config.min_pixels, config.max_pixels);
    } catch (const std::invalid_argument& error) {
        std::cerr << "[trtmc] Failed to smart-resize Qwen-VL image: " << error.what() << std::endl;
        return loaded;
    }
    auto resized = config.interpolation == "bicubic"
                       ? qwen_vl_resize_bicubic_antialias_u8(image.pixels.data(), image.width,
                                                             image.height, target[1], target[0])
                       : resize_raw(image.pixels.data(), image.width, image.height, target[1],
                                    target[0], resolve_stbir_filter(config.interpolation));
    if (resized.empty() ||
        !normalize_to_chw(resized, target[1], target[0], config, loaded.img_chw)) {
        std::cerr << "[trtmc] Failed to resize or normalize Qwen-VL image" << std::endl;
        return loaded;
    }
    loaded.target_height = target[0];
    loaded.target_width = target[1];
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// Center-crop to square, then resize + normalize.
static LoadedImage load_crop_resize_normalize(const runtime::adapters::io::DecodedImage& image,
                                              const QwenVlPreprocessConfig& config) {
    LoadedImage loaded;

    if (image.empty()) {
        std::cerr << "[trtmc] Failed to preprocess cropped image: decoded image missing"
                  << std::endl;
        return loaded;
    }

    // Center-crop to square
    const int crop_size = std::min(image.width, image.height);
    const int x_off = (image.width - crop_size) / 2;
    const int y_off = (image.height - crop_size) / 2;

    std::vector<unsigned char> cropped(static_cast<std::size_t>(crop_size) * crop_size * 3);
    for (int y = 0; y < crop_size; ++y) {
        const unsigned char* src_row =
            image.pixels.data() + (static_cast<std::size_t>(y + y_off) * image.width + x_off) * 3;
        unsigned char* dst_row = cropped.data() + static_cast<std::size_t>(y) * crop_size * 3;
        std::memcpy(dst_row, src_row, static_cast<std::size_t>(crop_size) * 3);
    }

    const int dst_h = target_height(config);
    const int dst_w = target_width(config);
    auto resized = resize_raw(cropped.data(), crop_size, crop_size, dst_w, dst_h,
                              resolve_stbir_filter(config.interpolation));
    if (resized.empty()) {
        std::cerr << "[trtmc] Failed to resize cropped image" << std::endl;
        return loaded;
    }

    if (!normalize_to_chw(resized, dst_w, dst_h, config, loaded.img_chw)) {
        std::cerr << "[trtmc] Failed to normalize cropped image" << std::endl;
        return loaded;
    }
    loaded.target_height = dst_h;
    loaded.target_width = dst_w;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// Aspect-ratio-preserving resize + zero-pad to square, then normalize.
static LoadedImage
load_aspect_preserve_resize_normalize(const runtime::adapters::io::DecodedImage& image,
                                      const QwenVlPreprocessConfig& config) {
    LoadedImage loaded;

    if (image.empty()) {
        std::cerr << "[trtmc] Failed to preprocess aspect-preserve image: decoded image missing"
                  << std::endl;
        return loaded;
    }

    const int dst_h = target_height(config);
    const int dst_w = target_width(config);
    const stbir_filter filter = resolve_stbir_filter(config.interpolation);

    // Compute scaled dimensions that fit inside the fixed profile.
    const float scale_w = static_cast<float>(dst_w) / static_cast<float>(image.width);
    const float scale_h = static_cast<float>(dst_h) / static_cast<float>(image.height);
    const float scale = std::min(scale_w, scale_h);
    const int new_w = std::max(1, static_cast<int>(image.width * scale));
    const int new_h = std::max(1, static_cast<int>(image.height * scale));

    // Resize preserving aspect ratio
    std::vector<unsigned char> resized_small(static_cast<std::size_t>(new_w) * new_h * 3);

    void* resize_result = stbir_resize(
        image.pixels.data(), image.width, image.height, image.width * 3, resized_small.data(),
        new_w, new_h, new_w * 3, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, filter);

    if (resize_result == nullptr) {
        std::cerr << "[trtmc] Failed to resize image (aspect-preserve)" << std::endl;
        return loaded;
    }

    // Zero-pad to the fixed profile (top-left aligned).
    std::vector<unsigned char> padded(static_cast<std::size_t>(dst_w) * dst_h * 3, 0);
    for (int y = 0; y < new_h; ++y) {
        const unsigned char* src_row =
            resized_small.data() + static_cast<std::size_t>(y) * new_w * 3;
        unsigned char* dst_row = padded.data() + static_cast<std::size_t>(y) * dst_w * 3;
        std::memcpy(dst_row, src_row, static_cast<std::size_t>(new_w) * 3);
    }

    if (!normalize_to_chw(padded, dst_w, dst_h, config, loaded.img_chw)) {
        std::cerr << "[trtmc] Failed to normalize aspect-preserve image" << std::endl;
        return loaded;
    }
    loaded.target_height = dst_h;
    loaded.target_width = dst_w;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// Aspect-ratio-preserving resize + center-pad with mean color, then normalize.
// Matches PIL ImageOps.pad(image, (size, size), color=mean*255).
static LoadedImage
load_pad_center_resize_normalize(const runtime::adapters::io::DecodedImage& image,
                                 const QwenVlPreprocessConfig& config) {
    LoadedImage loaded;

    if (image.empty()) {
        std::cerr << "[trtmc] Failed to preprocess pad-center image: decoded image missing"
                  << std::endl;
        return loaded;
    }

    const int dst_h = target_height(config);
    const int dst_w = target_width(config);
    const stbir_filter filter = resolve_stbir_filter(config.interpolation);

    // Compute scaled dimensions that fit inside the fixed profile.
    // This matches PIL ImageOps.pad behavior: scale to fit, then center.
    const float scale_w = static_cast<float>(dst_w) / static_cast<float>(image.width);
    const float scale_h = static_cast<float>(dst_h) / static_cast<float>(image.height);
    const float scale = std::min(scale_w, scale_h);
    const int new_w = std::max(1, static_cast<int>(image.width * scale));
    const int new_h = std::max(1, static_cast<int>(image.height * scale));

    // Resize preserving aspect ratio
    std::vector<unsigned char> resized_small(static_cast<std::size_t>(new_w) * new_h * 3);

    void* resize_result = stbir_resize(
        image.pixels.data(), image.width, image.height, image.width * 3, resized_small.data(),
        new_w, new_h, new_w * 3, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, filter);

    if (resize_result == nullptr) {
        std::cerr << "[trtmc] Failed to resize image (pad-center)" << std::endl;
        return loaded;
    }

    // Fill pad with mean color (mean * 255), matching ImageOps.pad color arg
    const unsigned char pad_r = static_cast<unsigned char>(config.image_mean[0] * 255.0F);
    const unsigned char pad_g = static_cast<unsigned char>(config.image_mean[1] * 255.0F);
    const unsigned char pad_b = static_cast<unsigned char>(config.image_mean[2] * 255.0F);

    std::vector<unsigned char> padded(static_cast<std::size_t>(dst_w) * dst_h * 3);
    for (std::size_t i = 0; i < padded.size(); i += 3) {
        padded[i + 0] = pad_r;
        padded[i + 1] = pad_g;
        padded[i + 2] = pad_b;
    }

    // Center the resized image in the padded canvas
    const int x_off = (dst_w - new_w) / 2;
    const int y_off = (dst_h - new_h) / 2;
    for (int y = 0; y < new_h; ++y) {
        const unsigned char* src_row =
            resized_small.data() + static_cast<std::size_t>(y) * new_w * 3;
        unsigned char* dst_row =
            padded.data() + (static_cast<std::size_t>(y + y_off) * dst_w + x_off) * 3;
        std::memcpy(dst_row, src_row, static_cast<std::size_t>(new_w) * 3);
    }

    if (!normalize_to_chw(padded, dst_w, dst_h, config, loaded.img_chw)) {
        std::cerr << "[trtmc] Failed to normalize pad-center image" << std::endl;
        return loaded;
    }
    loaded.target_height = dst_h;
    loaded.target_width = dst_w;
    loaded.channels = config.in_channels;
    loaded.ok = true;
    return loaded;
}

// ---------------------------------------------------------------------------
// Strategy: merge_group_chw
// ---------------------------------------------------------------------------

static QwenVlPreprocessedImage preprocess_merge_group_chw(const LoadedImage& loaded,
                                                          const QwenVlPreprocessConfig& config) {
    QwenVlPreprocessedImage result;
    ImageTransformParams params;
    params.layout = ImageTransformLayout::kMergeGroupChw;
    params.target_height = loaded.target_height;
    params.target_width = loaded.target_width;
    params.channels = loaded.channels;
    params.patch_size = config.patch_size;
    params.merge_size = config.merge_size;
    params.temporal_patch_size = config.temporal_patch_size;

    result.height = loaded.target_height;
    result.width = loaded.target_width;
    result.ok = transform_chw_layout(loaded.img_chw, params, result.pixel_values, result.channels);
    return result;
}

// ---------------------------------------------------------------------------
// Strategy: simple_chw
// ---------------------------------------------------------------------------

static QwenVlPreprocessedImage preprocess_simple_chw(const LoadedImage& loaded,
                                                     const QwenVlPreprocessConfig& config) {
    QwenVlPreprocessedImage result;
    ImageTransformParams params;
    params.layout = ImageTransformLayout::kSimpleChw;
    params.target_height = loaded.target_height;
    params.target_width = loaded.target_width;
    params.channels = loaded.channels;

    result.height = loaded.target_height;
    result.width = loaded.target_width;
    result.ok = transform_chw_layout(loaded.img_chw, params, result.pixel_values, result.channels);

    (void)config;
    return result;
}

// ---------------------------------------------------------------------------
// Strategy: patchify_chw
// ---------------------------------------------------------------------------

static QwenVlPreprocessedImage preprocess_patchify_chw(const LoadedImage& loaded,
                                                       const QwenVlPreprocessConfig& config) {
    QwenVlPreprocessedImage result;
    const int patch = config.patch_size;
    const int channels = loaded.channels;
    const int height = loaded.target_height;
    const int width = loaded.target_width;
    if (patch <= 0 || height % patch != 0 || width % patch != 0 || channels <= 0) {
        std::cerr << "[trtmc] Invalid patchify shape" << std::endl;
        return result;
    }

    const int grid_h = height / patch;
    const int grid_w = width / patch;
    const int num_patches = grid_h * grid_w;
    result.pixel_values.resize(static_cast<std::size_t>(num_patches) * channels * patch * patch);

    for (int gh = 0; gh < grid_h; ++gh) {
        for (int gw = 0; gw < grid_w; ++gw) {
            const int patch_idx = gh * grid_w + gw;
            for (int c = 0; c < channels; ++c) {
                for (int ph = 0; ph < patch; ++ph) {
                    for (int pw = 0; pw < patch; ++pw) {
                        const std::size_t dst =
                            (((static_cast<std::size_t>(patch_idx) * channels + c) * patch + ph) *
                                 patch +
                             pw);
                        const std::size_t src =
                            (static_cast<std::size_t>(c) * height + gh * patch + ph) * width +
                            gw * patch + pw;
                        result.pixel_values[dst] = loaded.img_chw[src];
                    }
                }
            }
        }
    }

    result.image_grid_hws = {grid_h, grid_w};
    result.channels = channels;
    result.height = height;
    result.width = width;
    result.ok = true;
    return result;
}

static std::vector<int32_t> build_dynamic_window_order(int32_t grid_h, int32_t grid_w,
                                                       const QwenVlPreprocessConfig& config,
                                                       QwenVlPreprocessedImage& result) {
    const int32_t merged_h = grid_h / config.merge_size;
    const int32_t merged_w = grid_w / config.merge_size;
    const int32_t window_groups = config.vision_window_size / config.merge_size / config.patch_size;
    const int32_t windows_h = (merged_h + window_groups - 1) / window_groups;
    const int32_t windows_w = (merged_w + window_groups - 1) / window_groups;
    const int32_t num_groups = merged_h * merged_w;
    result.vision_window_indices.reserve(static_cast<std::size_t>(num_groups));
    result.vision_reverse_indices.resize(static_cast<std::size_t>(num_groups));
    std::vector<int32_t> group_counts;
    group_counts.reserve(static_cast<std::size_t>(windows_h * windows_w));
    for (int32_t window_h = 0; window_h < windows_h; ++window_h) {
        for (int32_t window_w = 0; window_w < windows_w; ++window_w) {
            int32_t count = 0;
            for (int32_t local_h = 0; local_h < window_groups; ++local_h) {
                for (int32_t local_w = 0; local_w < window_groups; ++local_w) {
                    const int32_t row = window_h * window_groups + local_h;
                    const int32_t col = window_w * window_groups + local_w;
                    if (row >= merged_h || col >= merged_w)
                        continue;
                    const int32_t group = row * merged_w + col;
                    result.vision_reverse_indices[static_cast<std::size_t>(group)] =
                        static_cast<int32_t>(result.vision_window_indices.size());
                    result.vision_window_indices.push_back(group);
                    ++count;
                }
            }
            if (count > 0)
                group_counts.push_back(count);
        }
    }
    return group_counts;
}

static void build_dynamic_vision_rope(int32_t merged_w, const QwenVlPreprocessConfig& config,
                                      QwenVlPreprocessedImage& result) {
    const int32_t head_dim = config.vision_embed_dim / config.vision_num_heads;
    const int32_t half_head_dim = head_dim / 2;
    const int32_t axis_frequencies = half_head_dim / 2;
    const int32_t merge = config.merge_size;
    const std::size_t patch_count =
        result.vision_window_indices.size() * static_cast<std::size_t>(merge * merge);
    result.vision_rope_half_dim = half_head_dim;
    result.vision_cos_half.reserve(patch_count * static_cast<std::size_t>(half_head_dim));
    result.vision_sin_half.reserve(patch_count * static_cast<std::size_t>(half_head_dim));

    // RoPE values depend only on the spatial position and frequency. A large
    // document can contain tens of thousands of patches but only a few
    // hundred distinct row/column positions. Compute each position/frequency
    // pair once instead of repeating pow/cos/sin for every patch.
    int32_t max_group_h = -1;
    int32_t max_group_w = -1;
    for (const int32_t group : result.vision_window_indices) {
        max_group_h = std::max(max_group_h, group / merged_w);
        max_group_w = std::max(max_group_w, group % merged_w);
    }
    const int32_t position_count_h = (max_group_h + 1) * merge;
    const int32_t position_count_w = (max_group_w + 1) * merge;

    const auto build_axis_cache = [&](int32_t position_count) {
        std::pair<std::vector<float>, std::vector<float>> cache;
        cache.first.resize(static_cast<std::size_t>(position_count) * axis_frequencies);
        cache.second.resize(static_cast<std::size_t>(position_count) * axis_frequencies);
        for (int32_t position = 0; position < position_count; ++position) {
            for (int32_t frequency = 0; frequency < axis_frequencies; ++frequency) {
                const double exponent = static_cast<double>(2 * frequency) / half_head_dim;
                const double angle = position / std::pow(config.vision_rope_theta, exponent);
                const std::size_t offset =
                    static_cast<std::size_t>(position) * axis_frequencies + frequency;
                cache.first[offset] = static_cast<float>(std::cos(angle));
                cache.second[offset] = static_cast<float>(std::sin(angle));
            }
        }
        return cache;
    };
    const auto height_cache = build_axis_cache(position_count_h);
    const auto width_cache = build_axis_cache(position_count_w);

    for (const int32_t group : result.vision_window_indices) {
        const int32_t group_h = group / merged_w;
        const int32_t group_w = group % merged_w;
        for (int32_t merge_h = 0; merge_h < merge; ++merge_h) {
            for (int32_t merge_w = 0; merge_w < merge; ++merge_w) {
                const int32_t position_h = group_h * merge + merge_h;
                const int32_t position_w = group_w * merge + merge_w;
                for (const auto& [position, cache] :
                     {std::pair{position_h, &height_cache}, std::pair{position_w, &width_cache}}) {
                    const std::size_t begin = static_cast<std::size_t>(position) * axis_frequencies;
                    result.vision_cos_half.insert(result.vision_cos_half.end(),
                                                  cache->first.begin() + begin,
                                                  cache->first.begin() + begin + axis_frequencies);
                    result.vision_sin_half.insert(result.vision_sin_half.end(),
                                                  cache->second.begin() + begin,
                                                  cache->second.begin() + begin + axis_frequencies);
                }
            }
        }
    }
}

static void build_dynamic_window_padding(const std::vector<int32_t>& group_counts,
                                         const QwenVlPreprocessConfig& config,
                                         QwenVlPreprocessedImage& result) {
    const int32_t merge_unit = config.merge_size * config.merge_size;
    const int32_t window_groups = config.vision_window_size / config.merge_size / config.patch_size;
    const int32_t patches_per_window = window_groups * window_groups * merge_unit;
    result.vision_window_count = static_cast<int32_t>(group_counts.size());
    result.vision_patches_per_window = patches_per_window;
    int32_t compact_offset = 0;
    for (const int32_t groups : group_counts) {
        const int32_t real_patches = groups * merge_unit;
        const int32_t padded_offset =
            static_cast<int32_t>(result.vision_padded_window_indices.size());
        for (int32_t patch = 0; patch < patches_per_window; ++patch) {
            const bool real = patch < real_patches;
            result.vision_padded_window_indices.push_back(compact_offset + (real ? patch : 0));
            result.vision_window_mask.push_back(real ? 0.0F : -1.0e9F);
            if (real)
                result.vision_compact_window_indices.push_back(padded_offset + patch);
        }
        compact_offset += real_patches;
    }
}

static bool valid_dynamic_patch_config(const LoadedImage& loaded,
                                       const QwenVlPreprocessConfig& config) {
    return config.patch_size > 0 && config.merge_size > 0 && config.temporal_patch_size > 0 &&
           config.vision_num_heads > 0 && config.vision_embed_dim % config.vision_num_heads == 0 &&
           loaded.channels > 0;
}

static void copy_dynamic_patch(const LoadedImage& loaded, const QwenVlPreprocessConfig& config,
                               int32_t group_columns, int32_t group_h, int32_t group_w,
                               int32_t merge_h, int32_t merge_w, int32_t patch_vector,
                               std::vector<float>& destination) {
    const int32_t patch = config.patch_size;
    const int32_t merge = config.merge_size;
    const std::size_t patch_index =
        ((static_cast<std::size_t>(group_h) * group_columns + group_w) * merge + merge_h) * merge +
        merge_w;
    float* patch_destination = destination.data() + patch_index * patch_vector;
    for (int32_t channel = 0; channel < loaded.channels; ++channel) {
        for (int32_t temporal = 0; temporal < config.temporal_patch_size; ++temporal) {
            for (int32_t patch_h = 0; patch_h < patch; ++patch_h) {
                const int32_t source_h = (group_h * merge + merge_h) * patch + patch_h;
                const int32_t source_w = (group_w * merge + merge_w) * patch;
                const std::size_t source =
                    (static_cast<std::size_t>(channel) * loaded.target_height + source_h) *
                        loaded.target_width +
                    source_w;
                const std::size_t target =
                    ((static_cast<std::size_t>(channel) * config.temporal_patch_size + temporal) *
                         patch +
                     patch_h) *
                    patch;
                std::copy_n(loaded.img_chw.data() + source, patch, patch_destination + target);
            }
        }
    }
}

static QwenVlPreprocessedImage
preprocess_qwen_smart_resize_patchify(const LoadedImage& loaded,
                                      const QwenVlPreprocessConfig& config) {
    QwenVlPreprocessedImage result;
    const int32_t patch = config.patch_size;
    const int32_t merge = config.merge_size;
    if (!valid_dynamic_patch_config(loaded, config)) {
        std::cerr << "[trtmc] Invalid dynamic Qwen-VL patch configuration" << std::endl;
        return result;
    }
    const int32_t grid_h = loaded.target_height / patch;
    const int32_t grid_w = loaded.target_width / patch;
    if (grid_h % merge != 0 || grid_w % merge != 0) {
        std::cerr << "[trtmc] Invalid dynamic Qwen-VL patch configuration" << std::endl;
        return result;
    }

    const int32_t patch_vector = loaded.channels * config.temporal_patch_size * patch * patch;
    const int32_t num_patches = grid_h * grid_w;
    result.pixel_values.resize(static_cast<std::size_t>(num_patches) * patch_vector);
    const int32_t group_rows = grid_h / merge;
    const int32_t group_columns = grid_w / merge;
    parallel_for_ranges(group_rows, 2, [&](int32_t begin_group_h, int32_t end_group_h) {
        for (int32_t group_h = begin_group_h; group_h < end_group_h; ++group_h) {
            for (int32_t group_w = 0; group_w < group_columns; ++group_w) {
                for (int32_t merge_h = 0; merge_h < merge; ++merge_h) {
                    for (int32_t merge_w = 0; merge_w < merge; ++merge_w) {
                        copy_dynamic_patch(loaded, config, group_columns, group_h, group_w, merge_h,
                                           merge_w, patch_vector, result.pixel_values);
                    }
                }
            }
        }
    });

    result.image_grid_hws = {grid_h, grid_w};
    result.channels = patch_vector;
    result.height = loaded.target_height;
    result.width = loaded.target_width;
    const auto group_counts = build_dynamic_window_order(grid_h, grid_w, config, result);
    build_dynamic_vision_rope(grid_w / merge, config, result);
    build_dynamic_window_padding(group_counts, config, result);
    result.ok = true;
    return result;
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

using LoadImageFn = LoadedImage (*)(const runtime::adapters::io::DecodedImage& image,
                                    const QwenVlPreprocessConfig& config);

using PreprocessImageFn = QwenVlPreprocessedImage (*)(const LoadedImage& loaded,
                                                      const QwenVlPreprocessConfig& config);

struct PreprocessDispatch {
    LoadImageFn load_fn;
    PreprocessImageFn preprocess_fn;
    bool warn_unknown_type{false};
};

static PreprocessDispatch resolve_preprocess_dispatch(const std::string& preprocessor_type) {
    if (preprocessor_type == "qwen_smart_resize_patchify")
        return {load_smart_resize_normalize, preprocess_qwen_smart_resize_patchify, false};
    if (preprocessor_type == "aspect_preserve_merge_group_chw")
        return {load_aspect_preserve_resize_normalize, preprocess_merge_group_chw, false};
    if (preprocessor_type == "center_crop_chw")
        return {load_crop_resize_normalize, preprocess_simple_chw, false};
    if (preprocessor_type == "aspect_preserve_chw")
        return {load_aspect_preserve_resize_normalize, preprocess_simple_chw, false};
    if (preprocessor_type == "pad_center_chw")
        return {load_pad_center_resize_normalize, preprocess_simple_chw, false};
    if (preprocessor_type == "simple_chw")
        return {load_resize_normalize, preprocess_simple_chw, false};
    if (preprocessor_type == "patchify_chw")
        return {load_resize_normalize, preprocess_patchify_chw, false};

    const bool warn_unknown = (preprocessor_type != "merge_group_chw");
    return {load_resize_normalize, preprocess_merge_group_chw, warn_unknown};
}

static QwenVlPreprocessedImage
run_preprocess_dispatch(const runtime::adapters::io::DecodedImage& image,
                        const QwenVlPreprocessConfig& config, const PreprocessDispatch& dispatch) {
    LoadedImage loaded = dispatch.load_fn(image, config);
    if (!loaded.ok) {
        return QwenVlPreprocessedImage{};
    }
    return dispatch.preprocess_fn(loaded, config);
}

QwenVlPreprocessedImage
qwen_vl_preprocess_decoded_image(const runtime::adapters::io::DecodedImage& image,
                                 const QwenVlPreprocessConfig& config) {
    const auto dispatch = resolve_preprocess_dispatch(config.preprocessor_type);
    if (dispatch.warn_unknown_type) {
        std::cerr << "[trtmc] WARNING: Unknown preprocessor_type \"" << config.preprocessor_type
                  << "\", falling back to merge_group_chw" << std::endl;
    }

    return run_preprocess_dispatch(image, config, dispatch);
}

std::string qwen_vl_format_prompt(const std::string& user_prompt,
                                  const QwenVlPreprocessConfig& config, int32_t image_pad_tokens) {
    // Build image_pads string: repeat image_token_str num_image_pad_tokens times
    std::string image_pads;
    const int32_t pad_count =
        image_pad_tokens >= 0 ? image_pad_tokens : config.num_image_pad_tokens;
    image_pads.reserve(static_cast<std::size_t>(pad_count) * config.image_token_str.size());
    for (int32_t i = 0; i < pad_count; ++i) {
        image_pads += config.image_token_str;
    }

    // Replace {image_pads} and {prompt} in the template
    std::string result = config.vl_prompt_template;

    const std::string pads_placeholder = "{image_pads}";
    const std::size_t pads_pos = result.find(pads_placeholder);
    if (pads_pos != std::string::npos) {
        result.replace(pads_pos, pads_placeholder.size(), image_pads);
    }

    const std::string prompt_placeholder = "{prompt}";
    const std::size_t prompt_pos = result.find(prompt_placeholder);
    if (prompt_pos != std::string::npos) {
        result.replace(prompt_pos, prompt_placeholder.size(), user_prompt);
    }

    return result;
}

QwenVlMropePositions qwen_vl_build_mrope_positions(const std::vector<int32_t>& input_ids,
                                                   int32_t image_token_id,
                                                   int32_t num_image_features, int32_t grid_height,
                                                   int32_t grid_width) {
    QwenVlMropePositions result;
    result.token_positions.resize(input_ids.size());

    const auto fill_text_positions = [&result, &input_ids]() {
        for (std::size_t i = 0; i < input_ids.size(); ++i) {
            const int32_t position = static_cast<int32_t>(i);
            result.token_positions[i] = {position, position, position};
        }
        result.next_position = static_cast<int32_t>(input_ids.size());
    };

    const auto image_begin = std::find(input_ids.begin(), input_ids.end(), image_token_id);
    const bool valid_grid =
        grid_height > 0 && grid_width > 0 && grid_height * grid_width == num_image_features;
    if (image_begin == input_ids.end() || !valid_grid) {
        fill_text_positions();
        return result;
    }

    const std::size_t image_offset =
        static_cast<std::size_t>(std::distance(input_ids.begin(), image_begin));
    const std::size_t image_end = image_offset + static_cast<std::size_t>(num_image_features);
    if (image_end > input_ids.size()) {
        fill_text_positions();
        return result;
    }

    for (std::size_t i = 0; i < image_offset; ++i) {
        const int32_t position = static_cast<int32_t>(i);
        result.token_positions[i] = {position, position, position};
    }

    const int32_t vision_base = static_cast<int32_t>(image_offset);
    for (int32_t i = 0; i < num_image_features; ++i) {
        result.token_positions[image_offset + static_cast<std::size_t>(i)] = {
            vision_base, vision_base + i / grid_width, vision_base + i % grid_width};
    }

    int32_t text_position = vision_base + std::max(grid_height, grid_width);
    for (std::size_t i = image_end; i < input_ids.size(); ++i) {
        result.token_positions[i] = {text_position, text_position, text_position};
        ++text_position;
    }
    result.next_position = text_position;
    return result;
}

nlohmann::json parse_preprocess_document(const std::string& text) {
    auto document = nlohmann::json::parse(text);
    if (!document.is_object())
        throw std::runtime_error("runtime.json must be a JSON object");
    return document;
}

const nlohmann::json& require_preprocess_member(const nlohmann::json& document, const char* key) {
    const auto found = document.find(key);
    if (found == document.end())
        throw std::runtime_error(std::string("runtime.json is missing '") + key + "'");
    return *found;
}

int32_t require_preprocess_int(const nlohmann::json& document, const char* key) {
    const auto& value = require_preprocess_member(document, key);
    if (!value.is_number_integer() && !value.is_number_unsigned())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be an integer");
    return value.get<int32_t>();
}

float require_preprocess_number(const nlohmann::json& document, const char* key) {
    const auto& value = require_preprocess_member(document, key);
    if (!value.is_number())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be numeric");
    return value.get<float>();
}

std::string require_preprocess_string(const nlohmann::json& document, const char* key) {
    const auto& value = require_preprocess_member(document, key);
    if (!value.is_string())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be a string");
    return value.get<std::string>();
}

bool require_preprocess_bool(const nlohmann::json& document, const char* key) {
    const auto& value = require_preprocess_member(document, key);
    if (!value.is_boolean())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be a boolean");
    return value.get<bool>();
}

void require_preprocess_triplet(const nlohmann::json& document, const char* key,
                                float (&target)[3]) {
    const auto& value = require_preprocess_member(document, key);
    if (!value.is_array() || value.size() != 3)
        throw std::runtime_error(std::string("runtime.json '") + key + "' must have 3 numbers");
    for (std::size_t index = 0; index < 3; ++index) {
        if (!value[index].is_number())
            throw std::runtime_error(std::string("runtime.json '") + key + "' must have 3 numbers");
        target[index] = value[index].get<float>();
    }
}

QwenVlPreprocessConfig qwen_vl_parse_preprocess_config(const std::string& config_text) {
    const auto document = parse_preprocess_document(config_text);
    QwenVlPreprocessConfig cfg;
    cfg.preprocessor_type = require_preprocess_string(document, "preprocessor_type");
    cfg.image_token_id = require_preprocess_int(document, "image_token_id");
    cfg.fixed_image_size = require_preprocess_int(document, "fixed_image_size");
    cfg.fixed_image_height = require_preprocess_int(document, "fixed_image_height");
    cfg.fixed_image_width = require_preprocess_int(document, "fixed_image_width");
    cfg.patch_size = require_preprocess_int(document, "patch_size");
    cfg.merge_size = require_preprocess_int(document, "merge_size");
    cfg.temporal_patch_size = require_preprocess_int(document, "temporal_patch_size");
    cfg.num_image_pad_tokens = require_preprocess_int(document, "num_image_pad_tokens");
    cfg.vision_output_dim = require_preprocess_int(document, "vision_output_dim");
    cfg.vl_prompt_template = require_preprocess_string(document, "vl_prompt_template");
    cfg.image_token_str = require_preprocess_string(document, "image_token_str");
    cfg.interpolation = require_preprocess_string(document, "interpolation");
    require_preprocess_triplet(document, "image_mean", cfg.image_mean);
    require_preprocess_triplet(document, "image_std", cfg.image_std);
    cfg.dynamic_image_resolution = require_preprocess_bool(document, "dynamic_image_resolution");
    if (cfg.dynamic_image_resolution) {
        cfg.min_pixels = require_preprocess_int(document, "min_pixels");
        cfg.max_pixels = require_preprocess_int(document, "max_pixels");
        cfg.vision_embed_dim = require_preprocess_int(document, "vision_embed_dim");
        cfg.vision_num_heads = require_preprocess_int(document, "vision_num_heads");
        cfg.vision_window_size = require_preprocess_int(document, "vision_window_size");
        cfg.vision_rope_theta = require_preprocess_number(document, "vision_rope_theta");
    }
    if (cfg.fixed_image_size <= 0 || cfg.patch_size <= 0 || cfg.merge_size <= 0 ||
        cfg.temporal_patch_size <= 0 || cfg.vision_output_dim <= 0)
        throw std::runtime_error("runtime.json has invalid preprocessing geometry");
    if (cfg.interpolation != "nearest" && cfg.interpolation != "bilinear" &&
        cfg.interpolation != "bicubic")
        throw std::runtime_error("runtime.json has unsupported interpolation");
    return cfg;
}

} // namespace trtmc
