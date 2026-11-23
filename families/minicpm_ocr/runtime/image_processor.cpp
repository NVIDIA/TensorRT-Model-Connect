/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpm_ocr/runtime/image_processor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::minicpm_ocr {
namespace {

constexpr int32_t kChannels = 3;

// Python round() on a double: halfway cases go to the even neighbour. The
// default floating-point environment rounds to nearest-even.
int64_t round_half_even(double value) {
    return static_cast<int64_t>(std::nearbyint(value));
}

int32_t ensure_divide(double length, int32_t divisor) {
    const int64_t rounded = round_half_even(length / divisor) * divisor;
    const int64_t result = std::max<int64_t>(rounded, divisor);
    if (result > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("MiniCPM-OCR resized image dimension overflows");
    return static_cast<int32_t>(result);
}

// The uint8 antialiased bicubic resize (cubic kernel, taps, int16 weight
// precision, horizontal pass then vertical pass with 8-bit rounding between them)
// is adapted from PyTorch v2.15.0, aten/src/ATen/native/cpu/UpSampleKernel.cpp and
// UpSampleKernelAVXAntialias.h, BSD-3-Clause. PyTorch derives that code from
// Pillow-SIMD's Resample.c (MIT-CMU License). The notices are reproduced in
// families/minicpm_ocr/NOTICE.
double cubic(double x) {
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

struct AxisWeights {
    std::vector<int32_t> start;
    std::vector<int32_t> size;
    std::vector<int32_t> weights; // [out, max_size], int16 range
    int32_t max_size{0};
    int32_t precision{0};
};

// Weights of one antialiased bicubic axis, quantized to int16 with a precision
// chosen from the largest normalized weight of the whole axis.
AxisWeights axis_weights(int32_t in_size, int32_t out_size) {
    const double scale = static_cast<double>(in_size) / out_size;
    const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    AxisWeights axis;
    axis.max_size = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
    axis.start.resize(out_size);
    axis.size.resize(out_size);
    std::vector<double> real(static_cast<std::size_t>(out_size) * axis.max_size, 0.0);
    double max_weight = 0.0;
    for (int32_t i = 0; i < out_size; ++i) {
        const double center = scale * (i + 0.5);
        const auto low = std::max<int64_t>(static_cast<int64_t>(center - support + 0.5), 0);
        int64_t count =
            std::min<int64_t>(static_cast<int64_t>(center + support + 0.5), in_size) - low;
        count = std::clamp<int64_t>(count, 0, axis.max_size);
        double* row = real.data() + static_cast<std::size_t>(i) * axis.max_size;
        double total = 0.0;
        for (int64_t j = 0; j < count; ++j) {
            row[j] = cubic((static_cast<double>(j + low) - center + 0.5) * invscale);
            total += row[j];
        }
        if (total != 0.0) {
            for (int64_t j = 0; j < count; ++j) {
                row[j] /= total;
                max_weight = std::max(max_weight, row[j]);
            }
        }
        axis.start[i] = static_cast<int32_t>(low);
        axis.size[i] = static_cast<int32_t>(count);
    }
    int32_t precision = 0;
    for (; precision < 22; ++precision) {
        const int next = static_cast<int>(0.5 + max_weight * (1 << (precision + 1)));
        if (next >= (1 << 15))
            break;
    }
    axis.precision = precision;
    axis.weights.resize(real.size());
    for (std::size_t k = 0; k < real.size(); ++k) {
        const double value = real[k] * (1 << precision);
        axis.weights[k] = value < 0 ? static_cast<int>(value - 0.5) : static_cast<int>(value + 0.5);
    }
    return axis;
}

uint8_t clamp_shifted(int64_t accumulator, int32_t precision) {
    // Arithmetic shift, then saturate, as the reference integer kernels do.
    return static_cast<uint8_t>(std::clamp<int64_t>(accumulator >> precision, 0, 255));
}

std::vector<uint8_t> horizontal_pass(const std::vector<uint8_t>& image, int32_t height,
                                     int32_t width, int32_t out_width) {
    const auto axis = axis_weights(width, out_width);
    const int64_t bias = axis.precision > 0 ? (int64_t{1} << (axis.precision - 1)) : 0;
    std::vector<uint8_t> out(static_cast<std::size_t>(height) * out_width * kChannels);
    for (int32_t y = 0; y < height; ++y) {
        const uint8_t* row = image.data() + static_cast<std::size_t>(y) * width * kChannels;
        uint8_t* target = out.data() + static_cast<std::size_t>(y) * out_width * kChannels;
        for (int32_t x = 0; x < out_width; ++x) {
            const int32_t* w = axis.weights.data() + static_cast<std::size_t>(x) * axis.max_size;
            for (int32_t c = 0; c < kChannels; ++c) {
                int64_t accumulator = bias;
                for (int32_t j = 0; j < axis.size[x]; ++j)
                    accumulator +=
                        static_cast<int64_t>(
                            row[static_cast<std::size_t>(axis.start[x] + j) * kChannels + c]) *
                        w[j];
                target[static_cast<std::size_t>(x) * kChannels + c] =
                    clamp_shifted(accumulator, axis.precision);
            }
        }
    }
    return out;
}

std::vector<uint8_t> vertical_pass(const std::vector<uint8_t>& image, int32_t height, int32_t width,
                                   int32_t out_height) {
    const auto axis = axis_weights(height, out_height);
    const int64_t bias = axis.precision > 0 ? (int64_t{1} << (axis.precision - 1)) : 0;
    const std::size_t stride = static_cast<std::size_t>(width) * kChannels;
    std::vector<uint8_t> out(static_cast<std::size_t>(out_height) * stride);
    std::vector<int64_t> accumulator(stride);
    for (int32_t y = 0; y < out_height; ++y) {
        std::fill(accumulator.begin(), accumulator.end(), bias);
        const int32_t* w = axis.weights.data() + static_cast<std::size_t>(y) * axis.max_size;
        for (int32_t j = 0; j < axis.size[y]; ++j) {
            const uint8_t* row =
                image.data() + static_cast<std::size_t>(axis.start[y] + j) * stride;
            for (std::size_t k = 0; k < stride; ++k)
                accumulator[k] += static_cast<int64_t>(row[k]) * w[j];
        }
        uint8_t* target = out.data() + static_cast<std::size_t>(y) * stride;
        for (std::size_t k = 0; k < stride; ++k)
            target[k] = clamp_shifted(accumulator[k], axis.precision);
    }
    return out;
}

std::vector<uint8_t> rgb8(const internal::ImageView& image) {
    if (image.data == nullptr || image.height == 0 || image.width == 0 ||
        image.channels != kChannels || image.height > INT32_MAX || image.width > INT32_MAX ||
        (image.format != internal::ImageFormat::UInt8 &&
         image.format != internal::ImageFormat::Float32))
        throw std::invalid_argument(
            "MiniCPM-OCR requires a nonempty RGB image (UInt8 or Float32 HWC, 3 channels)");
    const uint64_t count = static_cast<uint64_t>(image.height) * image.width * kChannels;
    const std::size_t element = image.format == internal::ImageFormat::UInt8 ? 1 : sizeof(float);
    if (count > std::numeric_limits<std::size_t>::max() / element ||
        image.byte_size != count * element)
        throw std::invalid_argument("MiniCPM-OCR image storage does not match its dimensions");
    std::vector<uint8_t> result(static_cast<std::size_t>(count));
    if (image.format == internal::ImageFormat::UInt8) {
        std::memcpy(result.data(), image.data, result.size());
        return result;
    }
    const auto* bytes = static_cast<const unsigned char*>(image.data);
    for (std::size_t i = 0; i < result.size(); ++i) {
        float pixel;
        std::memcpy(&pixel, bytes + i * sizeof(float), sizeof(float));
        if (!std::isfinite(pixel) || pixel < 0.0F || pixel > 1.0F)
            throw std::invalid_argument("MiniCPM-OCR float pixels must be finite and in [0, 1]");
        result[i] = static_cast<uint8_t>(std::lround(pixel * 255.0F));
    }
    return result;
}

ImagePiece make_piece(const ImageProcessorConfig& c, const std::vector<uint8_t>& image,
                      int32_t image_width, int32_t top, int32_t left, int32_t height,
                      int32_t width) {
    const int32_t p = c.patch_size, m = c.merge_size;
    ImagePiece piece;
    piece.height = height / p;
    piece.width = width / p;
    const std::size_t count = static_cast<std::size_t>(piece.height) * piece.width;
    if (count > static_cast<std::size_t>(c.max_piece_patches))
        throw std::invalid_argument("MiniCPM-OCR image slice has " + std::to_string(count) +
                                    " patches, more than the vision engine profile allows (" +
                                    std::to_string(c.max_piece_patches) +
                                    "); the image aspect ratio is too extreme");
    const std::size_t width_values = static_cast<std::size_t>(kChannels) * p * p;
    piece.patches.resize(count * width_values);
    piece.position_ids.resize(count);
    const auto positions = nearest_position_ids(piece.height, piece.width, c.position_grid);
    std::size_t token = 0;
    for (int32_t block_y = 0; block_y < piece.height; block_y += m)
        for (int32_t block_x = 0; block_x < piece.width; block_x += m)
            for (int32_t dy = 0; dy < m; ++dy)
                for (int32_t dx = 0; dx < m; ++dx, ++token) {
                    const int32_t py = block_y + dy, px = block_x + dx;
                    piece.position_ids[token] =
                        positions[static_cast<std::size_t>(py) * piece.width + px];
                    float* out = piece.patches.data() + token * width_values;
                    for (int32_t channel = 0; channel < kChannels; ++channel) {
                        // (x - mean * 255) / (std * 255): the reference fuses
                        // rescale and normalize into one float32 subtract/divide.
                        const float mean = c.mean[channel] * 255.0F;
                        const float stddev = c.stddev[channel] * 255.0F;
                        for (int32_t i = 0; i < p; ++i) {
                            const std::size_t row =
                                (static_cast<std::size_t>(top + py * p + i) * image_width + left +
                                 static_cast<std::size_t>(px) * p) *
                                kChannels;
                            for (int32_t j = 0; j < p; ++j)
                                *out++ = (static_cast<float>(
                                              image[row + static_cast<std::size_t>(j) * kChannels +
                                                    channel]) -
                                          mean) /
                                         stddev;
                        }
                    }
                }
    return piece;
}

} // namespace

void ImageProcessorConfig::validate() const {
    if (patch_size <= 0 || merge_size != 2 || scale_resolution < patch_size * merge_size ||
        max_slice_nums < 1 || position_grid <= 0 || max_piece_patches < merge_size * merge_size ||
        max_piece_patches % (merge_size * merge_size) != 0)
        throw std::invalid_argument("Invalid MiniCPM-OCR image processor geometry");
    for (int32_t i = 0; i < kChannels; ++i)
        if (!std::isfinite(mean[i]) || !std::isfinite(stddev[i]) || stddev[i] <= 0.0F)
            throw std::invalid_argument("Invalid MiniCPM-OCR image normalization");
}

std::optional<SliceGrid> sliced_grid(int32_t height, int32_t width, int32_t max_slice_nums,
                                     int32_t scale_resolution) {
    const double log_ratio = std::log(static_cast<double>(width) / height);
    const double ratio = static_cast<double>(static_cast<int64_t>(width) * height) /
                         (static_cast<double>(scale_resolution) * scale_resolution);
    const int64_t multiple =
        std::min<int64_t>(static_cast<int64_t>(std::ceil(ratio)), max_slice_nums);
    if (multiple <= 1)
        return std::nullopt;
    SliceGrid best{1, 1};
    double min_error = std::numeric_limits<double>::infinity();
    for (const int64_t slices : {multiple - 1, multiple, multiple + 1}) {
        if (slices == 1 || slices > max_slice_nums)
            continue;
        for (int64_t rows = 1; rows <= slices; ++rows) {
            if (slices % rows != 0)
                continue;
            const int64_t cols = slices / rows;
            const double error = std::fabs(
                log_ratio - std::log(static_cast<double>(cols) / static_cast<double>(rows)));
            if (error < min_error) {
                best = {static_cast<int32_t>(rows), static_cast<int32_t>(cols)};
                min_error = error;
            } else if (error == min_error && rows > best.rows) {
                best = {static_cast<int32_t>(rows), static_cast<int32_t>(cols)};
            }
        }
    }
    return best;
}

std::array<int32_t, 2> best_resize(double height, double width, int32_t scale_resolution,
                                   int32_t multiple, bool allow_upscale) {
    if (height * width > static_cast<double>(scale_resolution) * scale_resolution ||
        allow_upscale) {
        const double aspect = width / height;
        const auto h = static_cast<int64_t>(scale_resolution / std::sqrt(aspect));
        const auto w = static_cast<int64_t>(static_cast<double>(h) * aspect);
        height = static_cast<double>(h);
        width = static_cast<double>(w);
    }
    return {ensure_divide(height, multiple), ensure_divide(width, multiple)};
}

std::array<int32_t, 2> refine_size(int32_t height, int32_t width, SliceGrid grid,
                                   int32_t scale_resolution, int32_t multiple) {
    const int32_t refine_width = ensure_divide(width, grid.cols);
    const int32_t refine_height = ensure_divide(height, grid.rows);
    const auto best = best_resize(static_cast<double>(refine_height) / grid.rows,
                                  static_cast<double>(refine_width) / grid.cols, scale_resolution,
                                  multiple, true);
    const int64_t out_height = static_cast<int64_t>(best[0]) * grid.rows;
    const int64_t out_width = static_cast<int64_t>(best[1]) * grid.cols;
    if (out_height > INT32_MAX || out_width > INT32_MAX)
        throw std::invalid_argument("MiniCPM-OCR refined image dimension overflows");
    return {static_cast<int32_t>(out_height), static_cast<int32_t>(out_width)};
}

std::vector<uint8_t> resize_rgb8(const std::vector<uint8_t>& image, int32_t height, int32_t width,
                                 int32_t out_height, int32_t out_width) {
    if (height <= 0 || width <= 0 || out_height <= 0 || out_width <= 0 ||
        image.size() != static_cast<std::size_t>(height) * width * kChannels)
        throw std::invalid_argument("Invalid MiniCPM-OCR resize geometry");
    std::vector<uint8_t> result = image;
    if (out_width != width)
        result = horizontal_pass(result, height, width, out_width);
    if (out_height != height)
        result = vertical_pass(result, height, out_width, out_height);
    return result;
}

std::vector<int32_t> nearest_position_ids(int32_t height, int32_t width, int32_t grid) {
    if (height <= 0 || width <= 0 || grid <= 0)
        throw std::invalid_argument("Invalid MiniCPM-OCR position grid");
    // The reference evaluates on the model device in float32: the boundaries
    // arange(1/grid, 1, 1/grid) as one fused multiply-add start + step * k, the
    // fractions arange(n) / n as k times the float32 reciprocal of n (scalar
    // division by multiplication); bucketize(right=True) then counts the
    // boundaries <= k / n. Exact ties make the rounding of both sides matter.
    const double start = 1.0 / grid;
    const auto count = static_cast<int64_t>(std::ceil((1.0 - start) / start));
    const auto step = static_cast<float>(start);
    std::vector<float> boundaries(static_cast<std::size_t>(std::max<int64_t>(count, 0)));
    for (std::size_t k = 0; k < boundaries.size(); ++k)
        boundaries[k] = std::fma(step, static_cast<float>(k), step);
    const auto bucket = [&](int32_t index, int32_t extent) {
        const float reciprocal = 1.0F / static_cast<float>(extent);
        const float x = static_cast<float>(index) * reciprocal;
        return static_cast<int32_t>(std::upper_bound(boundaries.begin(), boundaries.end(), x) -
                                    boundaries.begin());
    };
    std::vector<int32_t> rows(height), cols(width);
    for (int32_t i = 0; i < height; ++i)
        rows[i] = bucket(i, height);
    for (int32_t j = 0; j < width; ++j)
        cols[j] = bucket(j, width);
    std::vector<int32_t> result(static_cast<std::size_t>(height) * width);
    for (int32_t i = 0; i < height; ++i)
        for (int32_t j = 0; j < width; ++j)
            result[static_cast<std::size_t>(i) * width + j] = rows[i] * grid + cols[j];
    return result;
}

ProcessedImage preprocess_image(const ImageProcessorConfig& c, const internal::ImageView& view) {
    c.validate();
    const auto image = rgb8(view);
    const auto height = static_cast<int32_t>(view.height);
    const auto width = static_cast<int32_t>(view.width);
    const int32_t multiple = c.patch_size * c.merge_size;
    ProcessedImage result;
    const auto grid = c.slice_mode
                          ? sliced_grid(height, width, c.max_slice_nums, c.scale_resolution)
                          : std::nullopt;
    // Unsliced images are always resized to the slice resolution, larger or
    // smaller; the overview of a sliced image only ever shrinks.
    const auto overview =
        best_resize(height, width, c.scale_resolution, multiple, !grid.has_value());
    result.pieces.push_back(make_piece(c,
                                       resize_rgb8(image, height, width, overview[0], overview[1]),
                                       overview[1], 0, 0, overview[0], overview[1]));
    if (!grid)
        return result;
    result.grid = *grid;
    const auto refined = refine_size(height, width, *grid, c.scale_resolution, multiple);
    const auto pixels = resize_rgb8(image, height, width, refined[0], refined[1]);
    const int32_t slice_height = refined[0] / grid->rows;
    const int32_t slice_width = refined[1] / grid->cols;
    for (int32_t row = 0; row < grid->rows; ++row)
        for (int32_t col = 0; col < grid->cols; ++col)
            result.pieces.push_back(make_piece(c, pixels, refined[1], row * slice_height,
                                               col * slice_width, slice_height, slice_width));
    return result;
}

} // namespace trtmc::minicpm_ocr
