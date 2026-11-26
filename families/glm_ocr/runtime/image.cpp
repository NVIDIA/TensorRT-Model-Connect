/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/glm_ocr/runtime/image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::glm_ocr {
namespace {

// Python's round() of a float: half to even.
double round_half_even(double value) {
    const double floor = std::floor(value);
    const double diff = value - floor;
    if (diff > 0.5)
        return floor + 1.0;
    if (diff < 0.5)
        return floor;
    return std::fmod(floor, 2.0) == 0.0 ? floor : floor + 1.0;
}

// The resize below (cubic, make_filter, the two passes) is adapted from PyTorch's
// UpSampleKernel.cpp, which derives it from Pillow-SIMD; see the family NOTICE.
// PyTorch's antialiasing bicubic filter (a = -0.5, as Pillow).
double cubic(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2.0)
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
    return 0.0;
}

struct Filter {
    int taps{0};
    int precision{0};
    std::vector<int> start;
    std::vector<int> size;
    std::vector<std::int32_t> weights; // [out, taps], int16 range
};

// Index ranges and int16 weights as PyTorch's _compute_index_ranges_int16_weights.
Filter make_filter(int in_size, int out_size) {
    const double scale = static_cast<double>(in_size) / out_size;
    const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    Filter filter;
    filter.taps = static_cast<int>(std::ceil(support)) * 2 + 1;
    filter.start.resize(out_size);
    filter.size.resize(out_size);
    std::vector<double> weights(static_cast<std::size_t>(out_size) * filter.taps, 0.0);
    double largest = 0.0;
    for (int i = 0; i < out_size; ++i) {
        const double center = scale * (i + 0.5);
        const auto xmin =
            std::max<std::int64_t>(static_cast<std::int64_t>(center - support + 0.5), 0);
        auto xsize =
            std::min<std::int64_t>(static_cast<std::int64_t>(center + support + 0.5), in_size) -
            xmin;
        xsize = std::clamp<std::int64_t>(xsize, 0, filter.taps);
        double* row = weights.data() + static_cast<std::size_t>(i) * filter.taps;
        double total = 0.0;
        for (std::int64_t j = 0; j < xsize; ++j) {
            row[j] = cubic((static_cast<double>(j + xmin) - center + 0.5) * invscale);
            total += row[j];
        }
        if (total != 0.0) {
            for (std::int64_t j = 0; j < xsize; ++j) {
                row[j] /= total;
                largest = std::max(largest, row[j]);
            }
        }
        filter.start[i] = static_cast<int>(xmin);
        filter.size[i] = static_cast<int>(xsize);
    }
    for (filter.precision = 0; filter.precision < 22; ++filter.precision) {
        const int next = static_cast<int>(0.5 + largest * (1 << (filter.precision + 1)));
        if (next >= (1 << 15))
            break;
    }
    filter.weights.resize(weights.size());
    for (std::size_t k = 0; k < weights.size(); ++k) {
        const double value = weights[k] * (1 << filter.precision);
        filter.weights[k] =
            value < 0 ? static_cast<int>(-0.5 + value) : static_cast<int>(0.5 + value);
    }
    if (filter.precision < 1)
        throw std::runtime_error("glm_ocr resize filter has no fixed-point precision");
    return filter;
}

std::uint8_t clip8(std::int64_t value, int precision) {
    return static_cast<std::uint8_t>(std::clamp<std::int64_t>(value >> precision, 0, 255));
}

std::vector<std::uint8_t> resample_rows(const std::vector<std::uint8_t>& src, int height, int width,
                                        int out_width) {
    const auto filter = make_filter(width, out_width);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(height) * out_width * 3);
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* in_row = src.data() + static_cast<std::size_t>(y) * width * 3;
        for (int x = 0; x < out_width; ++x) {
            const std::int32_t* w =
                filter.weights.data() + static_cast<std::size_t>(x) * filter.taps;
            for (int c = 0; c < 3; ++c) {
                std::int64_t acc = std::int64_t{1} << (filter.precision - 1);
                for (int k = 0; k < filter.size[x]; ++k)
                    acc += static_cast<std::int64_t>(w[k]) * in_row[(filter.start[x] + k) * 3 + c];
                out[(static_cast<std::size_t>(y) * out_width + x) * 3 + c] =
                    clip8(acc, filter.precision);
            }
        }
    }
    return out;
}

std::vector<std::uint8_t> resample_columns(const std::vector<std::uint8_t>& src, int height,
                                           int width, int out_height) {
    const auto filter = make_filter(height, out_height);
    const std::size_t stride = static_cast<std::size_t>(width) * 3;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(out_height) * stride);
    for (int y = 0; y < out_height; ++y) {
        const std::int32_t* w = filter.weights.data() + static_cast<std::size_t>(y) * filter.taps;
        for (std::size_t x = 0; x < stride; ++x) {
            std::int64_t acc = std::int64_t{1} << (filter.precision - 1);
            for (int k = 0; k < filter.size[y]; ++k)
                acc += static_cast<std::int64_t>(w[k]) * src[(filter.start[y] + k) * stride + x];
            out[static_cast<std::size_t>(y) * stride + x] = clip8(acc, filter.precision);
        }
    }
    return out;
}

} // namespace

ImageSize smart_resize(int height, int width, const ImageConfig& config) {
    if (height <= 0 || width <= 0)
        throw std::invalid_argument("glm_ocr image must have positive height and width");
    const int factor = config.patch_size * config.merge_size;
    const std::int64_t frames = config.temporal_patch_size;
    std::int64_t h = height, w = width;
    if (h < factor || w < factor) {
        const double scale = std::max(static_cast<double>(factor) / static_cast<double>(h),
                                      static_cast<double>(factor) / static_cast<double>(w));
        h = static_cast<std::int64_t>(static_cast<double>(h) * scale);
        w = static_cast<std::int64_t>(static_cast<double>(w) * scale);
    }
    if (std::min(h, w) <= 0 ||
        static_cast<double>(std::max(h, w)) / static_cast<double>(std::min(h, w)) > 200.0)
        throw std::invalid_argument("glm_ocr image aspect ratio must be smaller than 200");
    const double hd = static_cast<double>(h), wd = static_cast<double>(w);
    auto h_bar = static_cast<std::int64_t>(round_half_even(hd / factor)) * factor;
    auto w_bar = static_cast<std::int64_t>(round_half_even(wd / factor)) * factor;
    const std::int64_t t_bar =
        static_cast<std::int64_t>(round_half_even(static_cast<double>(frames) / frames)) * frames;
    const std::int64_t pixels = t_bar * h_bar * w_bar;
    if (pixels > config.max_pixels) {
        const double beta =
            std::sqrt(static_cast<double>(frames * h * w) / static_cast<double>(config.max_pixels));
        h_bar = std::max<std::int64_t>(
            factor, static_cast<std::int64_t>(std::floor(hd / beta / factor)) * factor);
        w_bar = std::max<std::int64_t>(
            factor, static_cast<std::int64_t>(std::floor(wd / beta / factor)) * factor);
    } else if (pixels < config.min_pixels) {
        const double beta =
            std::sqrt(static_cast<double>(config.min_pixels) / static_cast<double>(frames * h * w));
        h_bar = static_cast<std::int64_t>(std::ceil(hd * beta / factor)) * factor;
        w_bar = static_cast<std::int64_t>(std::ceil(wd * beta / factor)) * factor;
    }
    if (h_bar > std::numeric_limits<int>::max() || w_bar > std::numeric_limits<int>::max())
        throw std::invalid_argument("glm_ocr resized image is too large");
    return {static_cast<int>(h_bar), static_cast<int>(w_bar)};
}

std::vector<std::uint8_t> to_rgb8(const internal::ImageView& image) {
    if (image.data == nullptr || image.height == 0 || image.width == 0)
        throw std::invalid_argument("glm_ocr image is empty");
    if (image.channels != 3)
        throw std::invalid_argument("glm_ocr accepts RGB images with exactly 3 channels");
    if (image.height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("glm_ocr image dimensions are too large");
    const std::uint64_t count = std::uint64_t{image.height} * image.width * 3;
    const bool uint8 = image.format == internal::ImageFormat::UInt8;
    if (!uint8 && image.format != internal::ImageFormat::Float32)
        throw std::invalid_argument("glm_ocr accepts UInt8 or Float32 RGB images");
    // Checked before allocating: a view whose dimensions exceed its buffer is refused, never
    // allocated for.
    const std::uint64_t element = uint8 ? 1 : sizeof(float);
    if (image.byte_size % element != 0 || image.byte_size / element != count)
        throw std::invalid_argument("glm_ocr image byte size does not match its dimensions");
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(count));
    if (uint8) {
        std::memcpy(pixels.data(), image.data, pixels.size());
        return pixels;
    }
    const auto* bytes = static_cast<const unsigned char*>(image.data);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        float value;
        std::memcpy(&value, bytes + i * sizeof(float), sizeof(float));
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            throw std::invalid_argument("glm_ocr float pixels must be finite and in [0, 1]");
        // The processor reads 8-bit images; float pixels are quantized to that grid.
        pixels[i] = static_cast<std::uint8_t>(std::lround(value * 255.0F));
    }
    return pixels;
}

std::vector<std::uint8_t> resize_bicubic(const std::vector<std::uint8_t>& rgb, int height,
                                         int width, int out_height, int out_width) {
    if (rgb.size() != static_cast<std::size_t>(height) * width * 3 || out_height <= 0 ||
        out_width <= 0)
        throw std::invalid_argument("glm_ocr resize received inconsistent sizes");
    std::vector<std::uint8_t> current = rgb;
    if (out_width != width)
        current = resample_rows(current, height, width, out_width);
    if (out_height != height)
        current = resample_columns(current, height, out_width, out_height);
    return current;
}

VisionInput preprocess_image(const internal::ImageView& image, const ImageConfig& config) {
    const auto rgb = to_rgb8(image);
    const int height = static_cast<int>(image.height), width = static_cast<int>(image.width);
    const auto size = smart_resize(height, width, config);
    const auto pixels = resize_bicubic(rgb, height, width, size.height, size.width);

    const int patch = config.patch_size, merge = config.merge_size,
              frames = config.temporal_patch_size;
    VisionInput input;
    input.grid_h = size.height / patch;
    input.grid_w = size.width / patch;
    const int count = input.num_patches();
    if (count > config.max_patches)
        throw std::invalid_argument("glm_ocr image resizes to " + std::to_string(size.height) +
                                    "x" + std::to_string(size.width) + " pixels (" +
                                    std::to_string(count) +
                                    " vision patches); this bundle was built for at most " +
                                    std::to_string(config.max_patches) +
                                    " (rebuild with larger --image-height/--image-width)");

    // (u8 - mean * 255) / (std * 255) in float32, as the processor's fused rescale+normalize.
    std::array<float, 3> shift{}, divisor{};
    for (int c = 0; c < 3; ++c) {
        shift[c] = config.mean[c] * 255.0F;
        divisor[c] = config.std[c] * 255.0F;
    }
    const std::size_t patch_dim = static_cast<std::size_t>(3) * frames * patch * patch;
    input.patches.resize(static_cast<std::size_t>(count) * patch_dim);
    const int half = config.head_dim / 2, quarter = config.head_dim / 4;
    input.rotary_cos.resize(static_cast<std::size_t>(count) * half);
    input.rotary_sin.resize(static_cast<std::size_t>(count) * half);
    std::vector<float> inv_freq(quarter);
    for (int j = 0; j < quarter; ++j) {
        const float exponent = static_cast<float>(2 * j) / static_cast<float>(half);
        inv_freq[j] = 1.0F / std::pow(config.rope_theta, exponent);
    }

    std::size_t index = 0;
    for (int bh = 0; bh < input.grid_h / merge; ++bh)
        for (int bw = 0; bw < input.grid_w / merge; ++bw)
            for (int mh = 0; mh < merge; ++mh)
                for (int mw = 0; mw < merge; ++mw, ++index) {
                    const int row = bh * merge + mh, col = bw * merge + mw;
                    float* out = input.patches.data() + index * patch_dim;
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < frames; ++t)
                            for (int py = 0; py < patch; ++py)
                                for (int px = 0; px < patch; ++px) {
                                    const int y = row * patch + py, x = col * patch + px;
                                    const float value = static_cast<float>(
                                        pixels[(static_cast<std::size_t>(y) * size.width + x) * 3 +
                                               c]);
                                    *out++ = (value - shift[c]) / divisor[c];
                                }
                    float* cos = input.rotary_cos.data() + index * half;
                    float* sin = input.rotary_sin.data() + index * half;
                    for (int j = 0; j < quarter; ++j) {
                        const float angle_h = static_cast<float>(row) * inv_freq[j];
                        const float angle_w = static_cast<float>(col) * inv_freq[j];
                        cos[j] = std::cos(angle_h);
                        sin[j] = std::sin(angle_h);
                        cos[quarter + j] = std::cos(angle_w);
                        sin[quarter + j] = std::sin(angle_w);
                    }
                }
    return input;
}

} // namespace trtmc::glm_ocr
