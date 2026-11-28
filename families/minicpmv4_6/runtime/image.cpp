/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Image slicing and resampling adapted from the MiniCPM-V 4.6 image processor
// of Hugging Face Transformers 5.22.2 (src/transformers/models/minicpmv4_6/
// image_processing_minicpmv4_6.py) and from the PyTorch antialiased uint8
// resampling it calls (aten/src/ATen/native/cpu/UpSampleKernel.cpp); see NOTICE.

#include "families/minicpmv4_6/runtime/image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace trtmc::minicpmv4_6 {
namespace {

// Python round(): ties to even.
std::int64_t round_half_even(double value) {
    const double floor = std::floor(value);
    const double fraction = value - floor;
    auto result = static_cast<std::int64_t>(floor);
    if (fraction > 0.5 || (fraction == 0.5 && (result & 1) != 0))
        ++result;
    return result;
}

// max(round(length / divisor) * divisor, divisor)
std::int64_t ensure_divide(double length, std::int64_t divisor) {
    return std::max<std::int64_t>(round_half_even(length / static_cast<double>(divisor)) * divisor,
                                  divisor);
}

// find_best_resize(): area-preserving resize to scale_resolution, then the
// 16x mode's divisibility (patch x 4 pixels).
std::pair<std::int64_t, std::int64_t> best_resize(double height, double width,
                                                  std::int32_t scale_resolution,
                                                  std::int32_t patch_size, bool allow_upscale) {
    const double limit = static_cast<double>(scale_resolution) * scale_resolution;
    if (height * width > limit || allow_upscale) {
        const double aspect_ratio = width / height;
        height = static_cast<double>(
            static_cast<std::int64_t>(scale_resolution / std::sqrt(aspect_ratio)));
        width = static_cast<double>(static_cast<std::int64_t>(height * aspect_ratio));
    }
    const std::int64_t divisor = static_cast<std::int64_t>(patch_size) * 4;
    return {ensure_divide(height, divisor), ensure_divide(width, divisor)};
}

// get_sliced_grid(): {rows, cols}, or {0, 0} for an image that is not sliced.
std::pair<std::int32_t, std::int32_t> sliced_grid(std::int32_t height, std::int32_t width,
                                                  std::int32_t max_slice_nums,
                                                  std::int32_t scale_resolution) {
    const double log_ratio = std::log(static_cast<double>(width) / height);
    const double ratio = static_cast<double>(width) * height /
                         (static_cast<double>(scale_resolution) * scale_resolution);
    // validate() bounds max_slice_nums, and with it this search; int64 keeps multiple + 1 in range.
    const auto multiple =
        static_cast<std::int64_t>(std::min<double>(std::ceil(ratio), max_slice_nums));
    if (multiple <= 1)
        return {0, 0};
    std::pair<std::int64_t, std::int64_t> best{1, 1};
    double min_error = std::numeric_limits<double>::infinity();
    for (const std::int64_t slices : {multiple - 1, multiple, multiple + 1}) {
        if (slices == 1 || slices > max_slice_nums)
            continue;
        for (std::int64_t rows = 1; rows <= slices; ++rows) {
            if (slices % rows != 0)
                continue;
            const std::int64_t cols = slices / rows;
            const double error = std::fabs(
                log_ratio - std::log(static_cast<double>(cols) / static_cast<double>(rows)));
            if (error < min_error) {
                best = {rows, cols};
                min_error = error;
            } else if (error == min_error && rows > best.first) {
                best = {rows, cols};
            }
        }
    }
    // Both factors divide a slice count that does not exceed max_slice_nums.
    return {static_cast<std::int32_t>(best.first), static_cast<std::int32_t>(best.second)};
}

std::int32_t checked_int(std::int64_t value, const char* what) {
    if (value <= 0 || value > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument(std::string("MiniCPM-V 4.6 image ") + what +
                                    " is out of range");
    return static_cast<std::int32_t>(value);
}

// Keys cubic convolution with a = -0.5.
double keys_cubic(double x) {
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0)
        return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2.0)
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
    return 0.0;
}

struct Taps {
    std::vector<std::int64_t> first, count;
    std::vector<std::int16_t> weights; // [output, width]
    std::int64_t width{0};
    unsigned precision{0};
};

Taps antialias_taps(std::int64_t input_size, std::int64_t output_size) {
    const double scale = static_cast<double>(input_size) / static_cast<double>(output_size);
    const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
    Taps taps;
    taps.width = static_cast<std::int64_t>(std::ceil(support)) * 2 + 1;
    taps.first.resize(static_cast<std::size_t>(output_size));
    taps.count.resize(static_cast<std::size_t>(output_size));
    std::vector<double> real(static_cast<std::size_t>(output_size * taps.width), 0.0);
    double max_weight = 0.0;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    for (std::int64_t i = 0; i < output_size; ++i) {
        const double center = scale * (static_cast<double>(i) + 0.5);
        const std::int64_t first =
            std::max(static_cast<std::int64_t>(center - support + 0.5), std::int64_t{0});
        std::int64_t count =
            std::min(static_cast<std::int64_t>(center + support + 0.5), input_size) - first;
        count = std::clamp(count, std::int64_t{0}, taps.width);
        double* row = real.data() + i * taps.width;
        double total = 0.0;
        for (std::int64_t j = 0; j < count; ++j) {
            row[j] = keys_cubic((static_cast<double>(j + first) - center + 0.5) * invscale);
            total += row[j];
        }
        if (total != 0.0) {
            for (std::int64_t j = 0; j < count; ++j) {
                row[j] /= total;
                max_weight = std::max(max_weight, row[j]);
            }
        }
        taps.first[static_cast<std::size_t>(i)] = first;
        taps.count[static_cast<std::size_t>(i)] = count;
    }
    for (taps.precision = 0; taps.precision < 22; ++taps.precision) {
        const int next = static_cast<int>(0.5 + max_weight * (1 << (taps.precision + 1)));
        if (next >= (1 << 15))
            break;
    }
    taps.weights.resize(real.size());
    for (std::size_t k = 0; k < real.size(); ++k) {
        const double value = real[k] * (1 << taps.precision);
        taps.weights[k] = static_cast<std::int16_t>(value < 0 ? static_cast<int>(-0.5 + value)
                                                              : static_cast<int>(0.5 + value));
    }
    return taps;
}

std::uint8_t clamp_pixel(int value, unsigned precision) {
    return static_cast<std::uint8_t>(std::clamp(value >> precision, 0, 255));
}

// The number of channel values of a nonempty RGB8 or RGB float view whose storage matches its
// int32 dimensions. The slice plan is computed from the dimensions, so they are checked first.
std::uint64_t checked_values(const internal::ImageView& image) {
    if (image.data == nullptr || image.height == 0 || image.width == 0 || image.channels != 3 ||
        (image.format != internal::ImageFormat::UInt8 &&
         image.format != internal::ImageFormat::Float32))
        throw std::invalid_argument("MiniCPM-V 4.6 requires nonempty RGB8 or RGB float images");
    constexpr auto kMaxSide = static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
    if (image.height > kMaxSide || image.width > kMaxSide)
        throw std::invalid_argument("MiniCPM-V 4.6 image dimensions are too large");
    // Both sides are below 2^31, so the count stays below 2^64.
    const std::uint64_t count = static_cast<std::uint64_t>(image.height) * image.width * 3;
    const std::size_t element = image.format == internal::ImageFormat::UInt8 ? 1 : sizeof(float);
    if (count > std::numeric_limits<std::size_t>::max() / element ||
        image.byte_size != count * element)
        throw std::invalid_argument("MiniCPM-V 4.6 RGB storage does not match image dimensions");
    return count;
}

std::vector<std::uint8_t> rgb8(const internal::ImageView& image) {
    const std::uint64_t count = checked_values(image);
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(count));
    if (image.format == internal::ImageFormat::UInt8) {
        std::memcpy(pixels.data(), image.data, pixels.size());
        return pixels;
    }
    const auto* source = static_cast<const std::uint8_t*>(image.data);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        float value;
        std::memcpy(&value, source + i * sizeof(float), sizeof(float));
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            throw std::invalid_argument("MiniCPM-V 4.6 RGB float pixels must be finite in [0, 1]");
        pixels[i] = static_cast<std::uint8_t>(std::lround(value * 255.0F));
    }
    return pixels;
}

// Normalized patches of one RGB8 crop, channel-major inside each patch as the
// reference packs them for its patch convolution.
std::vector<float> patchify(const ImageSettings& settings, const std::vector<std::uint8_t>& pixels,
                            std::int32_t image_width, std::int32_t top, std::int32_t left,
                            std::int32_t height, std::int32_t width) {
    const std::int32_t p = settings.patch_size;
    const std::int32_t rows = height / p, cols = width / p;
    const std::size_t patch_width = static_cast<std::size_t>(3) * p * p;
    // The reference fuses rescale (1 / 255) into mean and std in FP32.
    const float scale = static_cast<float>(1.0 / (1.0 / 255.0));
    std::array<float, 3> mean{}, stddev{};
    for (int c = 0; c < 3; ++c) {
        mean[c] = settings.mean[c] * scale;
        stddev[c] = settings.stddev[c] * scale;
    }
    std::vector<float> result(static_cast<std::size_t>(rows) * cols * patch_width);
    std::size_t out = 0;
    for (std::int32_t py = 0; py < rows; ++py)
        for (std::int32_t px = 0; px < cols; ++px)
            for (int c = 0; c < 3; ++c)
                for (std::int32_t y = 0; y < p; ++y)
                    for (std::int32_t x = 0; x < p; ++x) {
                        const std::size_t source =
                            (static_cast<std::size_t>(top + py * p + y) * image_width +
                             (left + px * p + x)) *
                                3 +
                            c;
                        result[out++] = (static_cast<float>(pixels[source]) - mean[c]) / stddev[c];
                    }
    return result;
}

VisionSlice make_slice(const ImageSettings& settings, const std::vector<std::uint8_t>& pixels,
                       std::int32_t image_width, std::int32_t top, std::int32_t left,
                       std::int32_t height, std::int32_t width) {
    VisionSlice slice;
    slice.height = height / settings.patch_size;
    slice.width = width / settings.patch_size;
    if (height % settings.patch_size || width % settings.patch_size || slice.height % 4 ||
        slice.width % 4)
        throw std::logic_error("MiniCPM-V 4.6 slice is not a whole number of 4 x 4 patch blocks");
    if (static_cast<std::int64_t>(slice.height) * slice.width > settings.max_patches)
        throw std::invalid_argument(
            "MiniCPM-V 4.6 image slice exceeds the vision engine's patch capacity; use a less "
            "elongated image");
    slice.patches = patchify(settings, pixels, image_width, top, left, height, width);
    slice.position_ids =
        bucket_position_ids(slice.height, slice.width, settings.positions_per_side);
    slice.window_index = block_order(slice.height, slice.width);
    slice.merge_index = block_order(slice.height / 2, slice.width / 2);
    return slice;
}

} // namespace

void ImageSettings::validate() const {
    if (patch_size <= 0 || positions_per_side <= 1 ||
        scale_resolution < std::int64_t{4} * patch_size || max_slice_nums <= 0 ||
        max_slice_nums > kMaxSliceNums || max_patches < 16)
        throw std::invalid_argument("Invalid MiniCPM-V 4.6 image settings");
    // A slice at the nominal resolution must fit the vision engine.
    const std::int64_t side = scale_resolution / patch_size;
    if (side * side > max_patches)
        throw std::invalid_argument(
            "MiniCPM-V 4.6 scale_resolution exceeds the vision engine's patch capacity");
    // patchify() normalizes 0..255 pixels as (pixel - mean * 255) / (std * 255) in FP32; both
    // ends must stay finite and inside FP16, the narrowest type of the vision engine.
    constexpr float kFp16Max = 65504.0F;
    for (int c = 0; c < 3; ++c) {
        const float offset = mean[c] * 255.0F, scale = stddev[c] * 255.0F;
        if (!std::isfinite(mean[c]) || !std::isfinite(stddev[c]) || !std::isfinite(offset) ||
            !std::isfinite(scale) || !(scale > 0.0F))
            throw std::invalid_argument("Invalid MiniCPM-V 4.6 image normalization");
        for (const float pixel : {0.0F, 255.0F}) {
            const float value = (pixel - offset) / scale;
            if (!std::isfinite(value) || std::fabs(value) > kFp16Max)
                throw std::invalid_argument(
                    "MiniCPM-V 4.6 image normalization leaves the FP16 range");
        }
    }
}

SlicePlan plan_slices(const ImageSettings& settings, std::int32_t height, std::int32_t width) {
    settings.validate();
    if (height <= 0 || width <= 0)
        throw std::invalid_argument("MiniCPM-V 4.6 image dimensions must be positive");
    SlicePlan plan;
    const auto grid =
        sliced_grid(height, width, settings.max_slice_nums, settings.scale_resolution);
    const bool sliced = grid.first > 0;
    const auto source = best_resize(height, width, settings.scale_resolution, settings.patch_size,
                                    /*allow_upscale=*/!sliced);
    plan.source_height = checked_int(source.first, "height");
    plan.source_width = checked_int(source.second, "width");
    if (!sliced)
        return plan;
    plan.rows = grid.first;
    plan.cols = grid.second;
    // get_refine_size(): the grid divides the image before the per-slice resize.
    const std::int64_t refine_width = ensure_divide(width, plan.cols);
    const std::int64_t refine_height = ensure_divide(height, plan.rows);
    const auto cell = best_resize(static_cast<double>(refine_height) / plan.rows,
                                  static_cast<double>(refine_width) / plan.cols,
                                  settings.scale_resolution, settings.patch_size,
                                  /*allow_upscale=*/true);
    plan.refine_height = checked_int(cell.first * plan.rows, "slice height");
    plan.refine_width = checked_int(cell.second * plan.cols, "slice width");
    return plan;
}

std::vector<std::uint8_t> resize_rgb8(const std::uint8_t* pixels, std::int32_t height,
                                      std::int32_t width, std::int32_t out_height,
                                      std::int32_t out_width) {
    if (height <= 0 || width <= 0 || out_height <= 0 || out_width <= 0)
        throw std::invalid_argument("MiniCPM-V 4.6 resize dimensions must be positive");
    // Each pass reads its input in place; only a resize of neither axis copies the source.
    const std::uint8_t* current = pixels;
    std::vector<std::uint8_t> horizontal;
    std::int32_t current_width = width;
    if (out_width != width) {
        const auto taps = antialias_taps(width, out_width);
        std::vector<std::uint8_t> next(static_cast<std::size_t>(height) * out_width * 3);
        for (std::int32_t y = 0; y < height; ++y)
            for (std::int32_t x = 0; x < out_width; ++x) {
                const auto first = taps.first[static_cast<std::size_t>(x)];
                const auto count = taps.count[static_cast<std::size_t>(x)];
                const std::int16_t* weights = taps.weights.data() + x * taps.width;
                for (int c = 0; c < 3; ++c) {
                    int sum = 1 << (taps.precision - 1);
                    for (std::int64_t j = 0; j < count; ++j)
                        sum += current[(static_cast<std::size_t>(y) * width + first + j) * 3 + c] *
                               weights[j];
                    next[(static_cast<std::size_t>(y) * out_width + x) * 3 + c] =
                        clamp_pixel(sum, taps.precision);
                }
            }
        horizontal = std::move(next);
        current = horizontal.data();
        current_width = out_width;
    }
    if (out_height != height) {
        const auto taps = antialias_taps(height, out_height);
        std::vector<std::uint8_t> next(static_cast<std::size_t>(out_height) * current_width * 3);
        for (std::int32_t y = 0; y < out_height; ++y) {
            const auto first = taps.first[static_cast<std::size_t>(y)];
            const auto count = taps.count[static_cast<std::size_t>(y)];
            const std::int16_t* weights = taps.weights.data() + y * taps.width;
            for (std::int32_t x = 0; x < current_width; ++x)
                for (int c = 0; c < 3; ++c) {
                    int sum = 1 << (taps.precision - 1);
                    for (std::int64_t j = 0; j < count; ++j)
                        sum += current[((first + j) * current_width + x) * 3 + c] * weights[j];
                    next[(static_cast<std::size_t>(y) * current_width + x) * 3 + c] =
                        clamp_pixel(sum, taps.precision);
                }
        }
        return next;
    }
    if (current == pixels)
        return std::vector<std::uint8_t>(pixels,
                                         pixels + static_cast<std::size_t>(height) * width * 3);
    return horizontal;
}

std::vector<std::int32_t> bucket_position_ids(std::int32_t height, std::int32_t width,
                                              std::int32_t side) {
    if (height <= 0 || width <= 0 || side <= 1)
        throw std::invalid_argument("MiniCPM-V 4.6 position grid must be positive");
    // torch.arange(1 / side, 1.0, 1 / side) on the GPU: the size is computed in
    // double, each element as an FP32 fused multiply-add start + step * i.
    const double step = 1.0 / side;
    const auto count = static_cast<std::int64_t>(std::ceil((1.0 - step) / step));
    const float start = static_cast<float>(step);
    std::vector<float> boundaries(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i)
        boundaries[static_cast<std::size_t>(i)] = std::fmaf(start, static_cast<float>(i), start);
    // arange(n) / n on the GPU multiplies by the FP32 reciprocal; bucketize(right=True)
    // returns the number of boundaries not greater than the coordinate.
    auto buckets = [&](std::int32_t n) {
        const float reciprocal = 1.0F / static_cast<float>(n);
        std::vector<std::int32_t> result(static_cast<std::size_t>(n));
        for (std::int32_t k = 0; k < n; ++k) {
            const float coordinate = static_cast<float>(k) * reciprocal;
            result[static_cast<std::size_t>(k)] = static_cast<std::int32_t>(
                std::upper_bound(boundaries.begin(), boundaries.end(), coordinate) -
                boundaries.begin());
        }
        return result;
    };
    const auto rows = buckets(height), cols = buckets(width);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(height) * width);
    for (std::int32_t y = 0; y < height; ++y)
        for (std::int32_t x = 0; x < width; ++x)
            ids[static_cast<std::size_t>(y) * width + x] =
                rows[static_cast<std::size_t>(y)] * side + cols[static_cast<std::size_t>(x)];
    return ids;
}

std::vector<std::int32_t> block_order(std::int32_t height, std::int32_t width) {
    if (height <= 0 || width <= 0 || height % 2 || width % 2)
        throw std::invalid_argument("MiniCPM-V 4.6 block grid must be even and positive");
    std::vector<std::int32_t> order;
    order.reserve(static_cast<std::size_t>(height) * width);
    for (std::int32_t by = 0; by < height; by += 2)
        for (std::int32_t bx = 0; bx < width; bx += 2)
            for (std::int32_t dy = 0; dy < 2; ++dy)
                for (std::int32_t dx = 0; dx < 2; ++dx)
                    order.push_back((by + dy) * width + bx + dx);
    return order;
}

SlicePlan plan_image(const ImageSettings& settings, const internal::ImageView& image) {
    checked_values(image);
    return plan_slices(settings, static_cast<std::int32_t>(image.height),
                       static_cast<std::int32_t>(image.width));
}

std::int64_t planned_rows(const ImageSettings& settings, const SlicePlan& plan) {
    // One vision row per 4 x 4 block of patches (VisionSlice::tokens()).
    const auto rows = [&](std::int64_t height, std::int64_t width) {
        return (height / settings.patch_size) * (width / settings.patch_size) / 16;
    };
    std::int64_t total = rows(plan.source_height, plan.source_width);
    if (plan.rows > 0)
        total += static_cast<std::int64_t>(plan.rows) * plan.cols *
                 rows(plan.refine_height / plan.rows, plan.refine_width / plan.cols);
    return total;
}

PreparedImage prepare_image(const ImageSettings& settings, const internal::ImageView& image) {
    const auto height = static_cast<std::int32_t>(image.height);
    const auto width = static_cast<std::int32_t>(image.width);
    const auto plan = plan_image(settings, image);
    // Check every planned slice before any resize allocates it.
    const auto check_patches = [&](std::int64_t slice_height, std::int64_t slice_width) {
        if ((slice_height / settings.patch_size) * (slice_width / settings.patch_size) >
            settings.max_patches)
            throw std::invalid_argument(
                "MiniCPM-V 4.6 image slice exceeds the vision engine's patch capacity; use a less "
                "elongated image");
    };
    check_patches(plan.source_height, plan.source_width);
    if (plan.rows > 0)
        check_patches(plan.refine_height / plan.rows, plan.refine_width / plan.cols);
    // Only an image whose slices fit is read: RGB8 in place, float pixels after conversion.
    std::vector<std::uint8_t> converted;
    const std::uint8_t* pixels = static_cast<const std::uint8_t*>(image.data);
    if (image.format != internal::ImageFormat::UInt8) {
        converted = rgb8(image);
        pixels = converted.data();
    }
    PreparedImage result;
    result.rows = plan.rows;
    result.cols = plan.cols;
    const auto source = resize_rgb8(pixels, height, width, plan.source_height, plan.source_width);
    result.slices.push_back(make_slice(settings, source, plan.source_width, 0, 0,
                                       plan.source_height, plan.source_width));
    if (plan.rows == 0)
        return result;
    const auto refined = resize_rgb8(pixels, height, width, plan.refine_height, plan.refine_width);
    const std::int32_t slice_height = plan.refine_height / plan.rows;
    const std::int32_t slice_width = plan.refine_width / plan.cols;
    for (std::int32_t row = 0; row < plan.rows; ++row)
        for (std::int32_t col = 0; col < plan.cols; ++col)
            result.slices.push_back(make_slice(settings, refined, plan.refine_width,
                                               row * slice_height, col * slice_width, slice_height,
                                               slice_width));
    return result;
}

std::string image_placeholder(const PreparedImage& image, std::int32_t image_index) {
    if (image.slices.empty() ||
        image.slices.size() != 1 + static_cast<std::size_t>(image.rows) * image.cols)
        throw std::logic_error("MiniCPM-V 4.6 prepared image has an inconsistent slice grid");
    auto pads = [](std::int32_t count) {
        std::string text;
        text.reserve(static_cast<std::size_t>(count) * 13);
        for (std::int32_t i = 0; i < count; ++i)
            text += "<|image_pad|>";
        return text;
    };
    std::string text = "<image_id>" + std::to_string(image_index) + "</image_id><image>" +
                       pads(image.slices.front().tokens()) + "</image>";
    for (std::int32_t row = 0; row < image.rows; ++row) {
        if (row > 0)
            text += "\n";
        for (std::int32_t col = 0; col < image.cols; ++col)
            text +=
                "<slice>" +
                pads(image.slices[1 + static_cast<std::size_t>(row) * image.cols + col].tokens()) +
                "</slice>";
    }
    return text;
}

} // namespace trtmc::minicpmv4_6
