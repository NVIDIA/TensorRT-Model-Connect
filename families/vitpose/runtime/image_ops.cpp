/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The box encoding, the unbiased warp and keypoints_from_heatmaps follow the VitPose image
// processor of Hugging Face Transformers 5.2.0; the bilinear sampling of 8-bit pixels and the
// Gaussian smoothing follow scipy 1.18.1 (scipy.ndimage, ni_interpolation.c, ni_splines.c and
// ni_filters.c), whose numerics that processor inherits; the sum that normalizes the Gaussian
// kernel follows NumPy's pairwise summation (loops_utils.h.src). Adapted code; see NOTICE.

#include "families/vitpose/runtime/image_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::vitpose {
namespace {

// The reference smooths every heatmap with sigma 0.8 before the DARK step.
constexpr double kDarkSigma = 0.8;
// numpy sums at most this many values with one unrolled pass (its pairwise block size).
constexpr std::size_t kPairwiseBlock = 128;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(std::string("vitpose ") + message);
}

// Converting a double beyond float's range to float is undefined; every such narrowing is checked
// in double first.
bool fits_float(double value) {
    return std::isfinite(value) &&
           std::fabs(value) <= static_cast<double>(std::numeric_limits<float>::max());
}

float to_float(double value, const char* message) {
    require(fits_float(value), message);
    return static_cast<float>(value);
}

void validate(const PreprocessConfig& config) {
    require(config.height > 1 && config.width > 1, "model input must be at least 2 x 2 pixels");
    require(std::isfinite(config.padding_factor) && config.padding_factor > 0.0,
            "box padding must be positive and finite");
    require(std::isfinite(config.rescale_factor) && config.rescale_factor > 0.0,
            "rescale factor must be positive and finite");
}

// One axis of scipy's order-1 spline sampling in mode "constant": a coordinate outside
// [0, length - 1] takes the constant zero; an inside one blends its floor and the next sample,
// and at the last sample the partner index is mirrored with weight zero.
struct AxisSample {
    bool inside{false};
    std::int32_t first{0};
    std::int32_t second{0};
    double first_weight{0.0};
    double second_weight{0.0};
};

AxisSample sample_axis(double coordinate, std::int32_t length) {
    AxisSample sample;
    if (!(coordinate >= 0.0) || coordinate > static_cast<double>(length - 1))
        return sample;
    const double start = std::floor(coordinate);
    sample.inside = true;
    sample.first = static_cast<std::int32_t>(start);
    sample.first_weight = 1.0 - (coordinate - start);
    // scipy derives the last weight from the others so that they sum to one.
    sample.second_weight = 1.0 - sample.first_weight;
    sample.second = sample.first + 1 < length ? sample.first + 1 : std::max(length - 2, 0);
    return sample;
}

// scipy stores an interpolated value into an unsigned 8-bit output by adding one half and
// truncating, after clamping into the type's range.
std::uint8_t store_u8(double value) {
    value = value > 0.0 ? value + 0.5 : 0.0;
    value = value > 255.0 ? 255.0 : value;
    return static_cast<std::uint8_t>(value);
}

// numpy's pairwise summation of a contiguous float64 array, for arrays of one block at most.
double numpy_sum(const std::vector<double>& values) {
    require(values.size() <= kPairwiseBlock, "smoothing kernel is too large");
    if (values.size() < 8) {
        double total = 0.0;
        for (const double value : values)
            total += value;
        return total;
    }
    double partial[8];
    for (std::size_t lane = 0; lane < 8; ++lane)
        partial[lane] = values[lane];
    std::size_t index = 8;
    for (; index < values.size() - values.size() % 8; index += 8) {
        for (std::size_t lane = 0; lane < 8; ++lane)
            partial[lane] += values[index + lane];
    }
    double total = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                   ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for (; index < values.size(); ++index)
        total += values[index];
    return total;
}

// scipy.ndimage._gaussian_kernel1d(0.8, 0, radius): exp(-0.5 / sigma^2 * x^2), normalized.
std::vector<double> gaussian_taps(std::int32_t radius) {
    const double factor = -0.5 / (kDarkSigma * kDarkSigma);
    std::vector<double> taps(static_cast<std::size_t>(2 * radius + 1));
    for (std::int32_t index = 0; index < 2 * radius + 1; ++index) {
        const auto offset = static_cast<std::int64_t>(index - radius);
        taps[static_cast<std::size_t>(index)] =
            std::exp(factor * static_cast<double>(offset * offset));
    }
    const double total = numpy_sum(taps);
    for (double& tap : taps)
        tap /= total;
    return taps;
}

// Mode "reflect" (d c b a | a b c d | d c b a): the index of a sample outside the line.
std::int32_t reflect(std::int64_t index, std::int32_t length) {
    const std::int64_t period = 2 * static_cast<std::int64_t>(length);
    std::int64_t folded = index % period;
    if (folded < 0)
        folded += period;
    return static_cast<std::int32_t>(folded < length ? folded : period - 1 - folded);
}

// scipy.ndimage.gaussian_filter(axes=(0, 1), mode="reflect") at one cell. scipy filters every
// column (axis 0), stores the result as float32, then filters every row (axis 1) in double and
// stores float32; for each kernel tap it adds the two mirrored samples first. Only the cells the
// DARK step reads are needed, and each is computed with exactly that arithmetic.
float smoothed_cell(const float* map, std::int32_t height, std::int32_t width, std::int32_t x,
                    std::int32_t y, const std::vector<double>& taps) {
    const auto radius = static_cast<std::int32_t>(taps.size() / 2);
    const auto center_tap = taps[static_cast<std::size_t>(radius)];
    const auto column_pass = [&](std::int64_t column) {
        const auto at = [&](std::int64_t row) {
            return static_cast<double>(map[static_cast<std::size_t>(reflect(row, height)) * width +
                                           static_cast<std::size_t>(reflect(column, width))]);
        };
        double value = at(y) * center_tap;
        for (std::int32_t tap = -radius; tap < 0; ++tap)
            value += (at(y + tap) + at(y - tap)) * taps[static_cast<std::size_t>(radius + tap)];
        return static_cast<double>(static_cast<float>(value));
    };
    double value = column_pass(x) * center_tap;
    for (std::int32_t tap = -radius; tap < 0; ++tap)
        value += (column_pass(x + tap) + column_pass(x - tap)) *
                 taps[static_cast<std::size_t>(radius + tap)];
    return static_cast<float>(value);
}

} // namespace

CropGeometry crop_geometry(const PersonBox& box, const PreprocessConfig& config) {
    validate(config);
    require(std::isfinite(box.x) && std::isfinite(box.y) && std::isfinite(box.width) &&
                std::isfinite(box.height) && box.width > 0.0 && box.height > 0.0,
            "person box must be finite with a positive width and height");
    // box_to_center_and_scale on Python floats: the centre keeps the box's own size, then the box
    // grows to the model's aspect ratio and is padded; numpy stores both pairs as float32.
    const double aspect = static_cast<double>(config.width) / static_cast<double>(config.height);
    double width = box.width;
    double height = box.height;
    CropGeometry geometry;
    geometry.center = {to_float(box.x + box.width * 0.5, "person box centre must fit float32"),
                       to_float(box.y + box.height * 0.5, "person box centre must fit float32")};
    if (width > aspect * height)
        height = width * 1.0 / aspect;
    else if (width < aspect * height)
        width = height * aspect;
    const auto padding = to_float(config.padding_factor, "box padding must fit float32");
    geometry.scale = {to_float(width / kPixelStd, "person box must fit float32") * padding,
                      to_float(height / kPixelStd, "person box must fit float32") * padding};
    for (const float value : {geometry.center[0], geometry.center[1]})
        require(std::isfinite(value), "person box centre must fit float32");
    for (const float value : {geometry.scale[0], geometry.scale[1]})
        require(std::isfinite(value) && value > 0.0F, "person box scale must fit float32");
    return geometry;
}

std::vector<std::uint8_t> warp_crop(const std::uint8_t* rgb, std::int32_t height,
                                    std::int32_t width, const CropGeometry& geometry,
                                    const PreprocessConfig& config) {
    validate(config);
    require(rgb != nullptr && height > 0 && width > 0, "source image must be non-empty");
    // get_warp_matrix(0, center * 2, size - 1, scale * kPixelStd): the sizes are float32, the
    // two scales float64, and the matrix is stored as float32.
    const float input_x = geometry.center[0] * 2.0F;
    const float input_y = geometry.center[1] * 2.0F;
    const float target_x = geometry.scale[0] * static_cast<float>(kPixelStd);
    const float target_y = geometry.scale[1] * static_cast<float>(kPixelStd);
    require(std::isfinite(input_x) && std::isfinite(input_y) && std::isfinite(target_x) &&
                std::isfinite(target_y) && target_x > 0.0F && target_y > 0.0F,
            "crop geometry must be finite with a positive scale");
    const double scale_x =
        (static_cast<double>(config.width) - 1.0) / static_cast<double>(target_x);
    const double scale_y =
        (static_cast<double>(config.height) - 1.0) / static_cast<double>(target_y);
    const float shift_x = -0.5F * input_x + 0.5F * target_x;
    const float shift_y = -0.5F * input_y + 0.5F * target_y;
    const auto matrix_x = to_float(scale_x, "crop warp must fit float32");
    const auto matrix_y = to_float(scale_y, "crop warp must fit float32");
    const auto offset_x =
        to_float(scale_x * static_cast<double>(shift_x), "crop warp must fit float32");
    const auto offset_y =
        to_float(scale_y * static_cast<double>(shift_y), "crop warp must fit float32");
    // scipy.linalg.inv of that matrix in float64: 1 / s on the diagonal and -(t * (1 / s)) in the
    // last column, as its LAPACK triangular inverse computes them.
    const double column_scale = 1.0 / static_cast<double>(matrix_x);
    const double column_shift = -(static_cast<double>(offset_x) * column_scale);
    const double row_scale = 1.0 / static_cast<double>(matrix_y);
    const double row_shift = -(static_cast<double>(offset_y) * row_scale);
    require(std::isfinite(column_scale) && std::isfinite(column_shift) &&
                std::isfinite(row_scale) && std::isfinite(row_shift),
            "crop warp must be finite");

    // affine_transform maps each output (row, column) to offset + matrix . (row, column); the
    // off-diagonal terms are zero, so every row and every column has one source coordinate.
    std::vector<AxisSample> columns(static_cast<std::size_t>(config.width));
    for (std::int32_t column = 0; column < config.width; ++column) {
        double coordinate = column_shift;
        coordinate += static_cast<double>(column) * column_scale;
        columns[static_cast<std::size_t>(column)] = sample_axis(coordinate, width);
    }
    std::vector<std::uint8_t> crop(static_cast<std::size_t>(config.height) * config.width * 3U);
    for (std::int32_t row = 0; row < config.height; ++row) {
        double coordinate = row_shift;
        coordinate += static_cast<double>(row) * row_scale;
        const auto source_row = sample_axis(coordinate, height);
        for (std::int32_t column = 0; column < config.width; ++column) {
            const auto& source_column = columns[static_cast<std::size_t>(column)];
            const auto target = (static_cast<std::size_t>(row) * config.width + column) * 3U;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                double value = 0.0;
                if (source_row.inside && source_column.inside) {
                    const auto pixel = [&](std::int32_t y, std::int32_t x) {
                        return static_cast<double>(
                            rgb[(static_cast<std::size_t>(y) * width + x) * 3U + channel]);
                    };
                    // scipy's filter order: (first, first), (first, second), (second, first),
                    // (second, second); each value times the row weight, then the column weight.
                    value += pixel(source_row.first, source_column.first) *
                             source_row.first_weight * source_column.first_weight;
                    value += pixel(source_row.first, source_column.second) *
                             source_row.first_weight * source_column.second_weight;
                    value += pixel(source_row.second, source_column.first) *
                             source_row.second_weight * source_column.first_weight;
                    value += pixel(source_row.second, source_column.second) *
                             source_row.second_weight * source_column.second_weight;
                }
                crop[target + channel] = store_u8(value);
            }
        }
    }
    return crop;
}

bool normalization_is_finite(const PreprocessConfig& config) {
    if (!(config.rescale_factor > 0.0) || !std::isfinite(config.rescale_factor) ||
        !fits_float(1.0 / config.rescale_factor))
        return false;
    const auto factor = static_cast<float>(1.0 / config.rescale_factor);
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const float mean = config.mean[channel] * factor;
        const float deviation = config.std[channel] * factor;
        if (!std::isfinite(mean) || !std::isfinite(deviation) || deviation == 0.0F)
            return false;
        for (int value = 0; value <= 255; ++value) {
            if (!std::isfinite((static_cast<float>(value) - mean) / deviation))
                return false;
        }
    }
    return true;
}

std::vector<float> normalize_crop(const std::vector<std::uint8_t>& crop,
                                  const PreprocessConfig& config) {
    validate(config);
    require(normalization_is_finite(config),
            "rescale factor, image mean and image std must normalize 8-bit values to finite "
            "float32 values");
    const auto plane = static_cast<std::size_t>(config.height) * config.width;
    require(crop.size() == plane * 3U, "crop size does not match the model input");
    // The fast processor folds the rescale into the statistics in float32 (mean * (1 / rescale)),
    // then subtracts and divides.
    const auto factor = static_cast<float>(1.0 / config.rescale_factor);
    std::vector<float> pixels(plane * 3U);
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const float mean = config.mean[channel] * factor;
        const float deviation = config.std[channel] * factor;
        for (std::size_t index = 0; index < plane; ++index) {
            pixels[channel * plane + index] =
                (static_cast<float>(crop[index * 3U + channel]) - mean) / deviation;
        }
    }
    return pixels;
}

std::vector<Keypoint> decode_heatmaps(const float* heatmaps, std::int32_t keypoints,
                                      std::int32_t height, std::int32_t width,
                                      const CropGeometry& geometry, std::int32_t kernel) {
    require(heatmaps != nullptr && keypoints > 0 && height > 1 && width > 1,
            "heatmaps must be non-empty with at least 2 x 2 values");
    require(kernel >= 3 && kernel % 2 == 1 && static_cast<std::size_t>(kernel) <= kPairwiseBlock,
            "DARK kernel must be an odd size from 3 to 127");
    for (const float value : {geometry.center[0], geometry.center[1]})
        require(std::isfinite(value), "crop centre must be finite");
    for (const float value : {geometry.scale[0], geometry.scale[1]})
        require(std::isfinite(value) && value > 0.0F, "crop scale must be positive and finite");
    const auto plane = static_cast<std::size_t>(height) * width;
    for (std::size_t index = 0; index < plane * static_cast<std::size_t>(keypoints); ++index)
        require(std::isfinite(heatmaps[index]), "heatmaps must be finite");

    const auto taps = gaussian_taps((kernel - 1) / 2);
    // transform_preds: the crop side in pixels, spread over the heatmap's (size - 1) intervals.
    const float side_x = geometry.scale[0] * static_cast<float>(kPixelStd);
    const float side_y = geometry.scale[1] * static_cast<float>(kPixelStd);
    const float step_x = side_x / static_cast<float>(width - 1);
    const float step_y = side_y / static_cast<float>(height - 1);
    const double epsilon = static_cast<double>(std::numeric_limits<float>::epsilon());

    std::vector<Keypoint> result(static_cast<std::size_t>(keypoints));
    for (std::int32_t keypoint = 0; keypoint < keypoints; ++keypoint) {
        const float* map = heatmaps + static_cast<std::size_t>(keypoint) * plane;
        // get_keypoint_predictions: the first maximum in row-major order and its value.
        std::size_t peak = 0;
        for (std::size_t index = 1; index < plane; ++index) {
            if (map[index] > map[peak])
                peak = index;
        }
        auto& output = result[static_cast<std::size_t>(keypoint)];
        output.score = map[peak];
        if (!(output.score > 0.0F))
            continue;
        output.located = true;
        const auto peak_x = static_cast<std::int32_t>(peak % static_cast<std::size_t>(width));
        const auto peak_y = static_cast<std::int32_t>(peak / static_cast<std::size_t>(width));

        // post_dark_unbiased_data_processing: the smoothed heatmap, clipped to [0.001, 50], in
        // log, read at the peak and its neighbours with edge padding, all in float32.
        const auto at = [&](std::int32_t x, std::int32_t y) {
            x = std::clamp(x, 0, width - 1);
            y = std::clamp(y, 0, height - 1);
            const float smoothed = smoothed_cell(map, height, width, x, y, taps);
            return std::log(std::min(std::max(smoothed, 0.001F), 50.0F));
        };
        const float center = at(peak_x, peak_y);
        const float right = at(peak_x + 1, peak_y);
        const float left = at(peak_x - 1, peak_y);
        const float below = at(peak_x, peak_y + 1);
        const float above = at(peak_x, peak_y - 1);
        const float below_right = at(peak_x + 1, peak_y + 1);
        const float above_left = at(peak_x - 1, peak_y - 1);
        const float dx = 0.5F * (right - left);
        const float dy = 0.5F * (below - above);
        const float dxx = right - 2.0F * center + left;
        const float dyy = below - 2.0F * center + above;
        const float dxy =
            0.5F * (below_right - right - below + center + center - left - above + above_left);
        // One Newton step with inv(hessian + float32 epsilon * I) in float64. The reference
        // raises on a singular matrix; here a step that is not finite leaves the integer peak.
        const double a = static_cast<double>(dxx) + epsilon;
        const double b = static_cast<double>(dxy);
        const double d = static_cast<double>(dyy) + epsilon;
        const double determinant = a * d - b * b;
        const double move_x =
            (d * static_cast<double>(dx) - b * static_cast<double>(dy)) / determinant;
        const double move_y =
            (a * static_cast<double>(dy) - b * static_cast<double>(dx)) / determinant;
        float x = static_cast<float>(peak_x);
        float y = static_cast<float>(peak_y);
        const double refined_x = static_cast<double>(x) - move_x;
        const double refined_y = static_cast<double>(y) - move_y;
        if (fits_float(refined_x) && fits_float(refined_y)) {
            x = static_cast<float>(refined_x);
            y = static_cast<float>(refined_y);
        }
        // transform_preds, unbiased: heatmap cell i sits at centre - side / 2 + i * side / (n - 1).
        output.x = x * step_x + geometry.center[0] - side_x * 0.5F;
        output.y = y * step_y + geometry.center[1] - side_y * 0.5F;
    }
    return result;
}

} // namespace trtmc::vitpose
