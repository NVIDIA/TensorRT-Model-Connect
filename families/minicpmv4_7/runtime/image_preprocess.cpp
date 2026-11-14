/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/minicpmv4_7/runtime/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace trtmc::minicpmv4_7 {
namespace {

// Python round(): halves to even, as std::nearbyint does in the default rounding mode.
int32_t ensure_divide(double length, int32_t divisor) {
    const double quotient = std::nearbyint(length / divisor);
    return std::max(static_cast<int32_t>(quotient) * divisor, divisor);
}

// ---------------------------------------------------------------------------
// torchvision resize(uint8, BICUBIC, antialias=True) on CPU: PyTorch's separable
// fixed-point path (horizontal pass, then vertical pass, each rounded to uint8).
// Adapted from PyTorch v2.12.0 UpSampleKernel.cpp (BSD-3-Clause, after Pillow-SIMD's
// Resample.c); the notices are in families/minicpmv4_7/NOTICE.
// ---------------------------------------------------------------------------
struct AntialiasWeights {
    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int16_t> values;
    int32_t stride{0};
    uint32_t precision{0};
};

double keys_cubic(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

AntialiasWeights antialias_weights(int32_t input_size, int32_t output_size) {
    constexpr int32_t interp_size = 4;
    const double scale = static_cast<double>(input_size) / output_size;
    const double support = scale >= 1.0 ? (interp_size * 0.5) * scale : interp_size * 0.5;
    const int32_t kernel_size = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
    AntialiasWeights result;
    // PyTorch pads each int16 row to a 32-bit multiple; the padding never contributes.
    result.stride = kernel_size;
    while (result.stride % 4 != 0)
        ++result.stride;
    result.starts.resize(static_cast<std::size_t>(output_size));
    result.sizes.resize(static_cast<std::size_t>(output_size));
    result.values.assign(static_cast<std::size_t>(output_size) * result.stride, 0);

    std::vector<double> floating(static_cast<std::size_t>(output_size) * kernel_size, 0.0);
    double maximum = 0.0;
    const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
    for (int32_t i = 0; i < output_size; ++i) {
        const double center = scale * (i + 0.5);
        const int32_t start = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        const int32_t size =
            std::clamp(std::min(static_cast<int32_t>(center + support + 0.5), input_size) - start,
                       0, kernel_size);
        result.starts[static_cast<std::size_t>(i)] = start;
        result.sizes[static_cast<std::size_t>(i)] = size;
        double* row = floating.data() + static_cast<std::size_t>(i) * kernel_size;
        double total = 0.0;
        for (int32_t j = 0; j < size; ++j) {
            row[j] = keys_cubic((j + start - center + 0.5) * inverse_scale);
            total += row[j];
        }
        if (total != 0.0) {
            for (int32_t j = 0; j < size; ++j) {
                row[j] /= total;
                maximum = std::max(maximum, row[j]);
            }
        }
    }
    // The greatest precision whose largest coefficient still fits in int16.
    for (; result.precision < 22; ++result.precision) {
        const int32_t next = static_cast<int32_t>(
            0.5 + maximum * static_cast<double>(uint32_t{1} << (result.precision + 1)));
        if (next >= (1 << 15))
            break;
    }
    const double multiplier = static_cast<double>(uint32_t{1} << result.precision);
    for (int32_t i = 0; i < output_size; ++i) {
        const double* source = floating.data() + static_cast<std::size_t>(i) * kernel_size;
        int16_t* destination = result.values.data() + static_cast<std::size_t>(i) * result.stride;
        for (int32_t j = 0; j < kernel_size; ++j) {
            const double value = source[j] * multiplier;
            destination[j] = static_cast<int16_t>(value < 0.0 ? static_cast<int32_t>(value - 0.5)
                                                              : static_cast<int32_t>(value + 0.5));
        }
    }
    return result;
}

uint8_t fixed_point(const uint8_t* source, std::size_t stride, const int16_t* weights,
                    int32_t count, uint32_t precision) {
    int32_t value = int32_t{1} << (precision - 1);
    for (int32_t j = 0; j < count; ++j)
        value += static_cast<int32_t>(source[static_cast<std::size_t>(j) * stride]) * weights[j];
    return static_cast<uint8_t>(std::clamp(value >> precision, 0, 255));
}

std::vector<uint8_t> resize_horizontal(const uint8_t* rgb, int32_t width, int32_t height,
                                       int32_t target_width) {
    const auto weights = antialias_weights(width, target_width);
    std::vector<uint8_t> out(static_cast<std::size_t>(height) * target_width * 3);
    for (int32_t y = 0; y < height; ++y)
        for (int32_t x = 0; x < target_width; ++x) {
            const auto start = weights.starts[static_cast<std::size_t>(x)];
            const auto* w = weights.values.data() + static_cast<std::size_t>(x) * weights.stride;
            for (int32_t c = 0; c < 3; ++c)
                out[(static_cast<std::size_t>(y) * target_width + x) * 3 + c] =
                    fixed_point(rgb + (static_cast<std::size_t>(y) * width + start) * 3 + c, 3, w,
                                weights.sizes[static_cast<std::size_t>(x)], weights.precision);
        }
    return out;
}

std::vector<uint8_t> resize_vertical(const uint8_t* rgb, int32_t width, int32_t height,
                                     int32_t target_height) {
    const auto weights = antialias_weights(height, target_height);
    std::vector<uint8_t> out(static_cast<std::size_t>(target_height) * width * 3);
    const auto row = static_cast<std::size_t>(width) * 3;
    for (int32_t y = 0; y < target_height; ++y) {
        const auto start = weights.starts[static_cast<std::size_t>(y)];
        const auto* w = weights.values.data() + static_cast<std::size_t>(y) * weights.stride;
        for (int32_t x = 0; x < width; ++x)
            for (int32_t c = 0; c < 3; ++c)
                out[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
                    fixed_point(rgb + (static_cast<std::size_t>(start) * width + x) * 3 + c, row, w,
                                weights.sizes[static_cast<std::size_t>(y)], weights.precision);
    }
    return out;
}

Crop make_crop(const std::vector<uint8_t>& rgb, int32_t image_width, int32_t top, int32_t left,
               int32_t height, int32_t width, const ProcessorConfig& config) {
    const int32_t p = config.patch_size;
    Crop crop;
    crop.grid_h = height / p;
    crop.grid_w = width / p;
    const auto count = static_cast<std::size_t>(crop.grid_h) * crop.grid_w;
    if (static_cast<int64_t>(count) > config.max_patches)
        throw std::invalid_argument("MiniCPM-V 4.7 image crop exceeds the vision engine capacity");
    // Fused rescale + normalize of the source processor: (u - 255 * mean) / (255 * std) in FP32.
    const float inverse_rescale = static_cast<float>(1.0 / (1.0 / 255.0));
    std::array<float, 3> mean{}, stddev{};
    for (int c = 0; c < 3; ++c) {
        mean[c] = config.mean[c] * inverse_rescale;
        stddev[c] = config.stddev[c] * inverse_rescale;
    }
    const auto width_values = static_cast<std::size_t>(3) * p * p;
    crop.patches.resize(count * width_values);
    for (int32_t gy = 0; gy < crop.grid_h; ++gy)
        for (int32_t gx = 0; gx < crop.grid_w; ++gx) {
            float* out = crop.patches.data() +
                         (static_cast<std::size_t>(gy) * crop.grid_w + gx) * width_values;
            for (int c = 0; c < 3; ++c)
                for (int32_t py = 0; py < p; ++py)
                    for (int32_t px = 0; px < p; ++px) {
                        const auto y = static_cast<std::size_t>(top + gy * p + py);
                        const auto x = static_cast<std::size_t>(left + gx * p + px);
                        const float value = rgb[(y * image_width + x) * 3 + c];
                        out[(static_cast<std::size_t>(c) * p + py) * p + px] =
                            (value - mean[c]) / stddev[c];
                    }
        }
    crop.position_ids = nearest_position_ids(crop.grid_h, crop.grid_w, config.patches_per_side);
    crop.window_index = block_order(crop.grid_h, crop.grid_w);
    crop.merge_index = block_order(crop.grid_h / 2, crop.grid_w / 2);
    return crop;
}

} // namespace

// sliced_grid, best_resize and refine_size port MiniCPMV4_6ImageProcessor (Transformers 5.22.2,
// Apache-2.0; see NOTICE).
std::array<int32_t, 2> sliced_grid(int32_t height, int32_t width, int32_t max_slice_nums,
                                   int32_t scale_resolution) {
    const double log_ratio = std::log(static_cast<double>(width) / height);
    const double ratio = static_cast<double>(width) * height /
                         (static_cast<double>(scale_resolution) * scale_resolution);
    const auto multiple = static_cast<int32_t>(
        std::min<double>(std::ceil(ratio), static_cast<double>(max_slice_nums)));
    if (multiple <= 1)
        return {0, 0};
    std::array<int32_t, 2> best{1, 1};
    double min_error = std::numeric_limits<double>::infinity();
    for (const int32_t slices : {multiple - 1, multiple, multiple + 1}) {
        if (slices == 1 || slices > max_slice_nums)
            continue;
        for (int32_t rows = 1; rows <= slices; ++rows) {
            if (slices % rows != 0)
                continue;
            const int32_t cols = slices / rows;
            const double error = std::abs(log_ratio - std::log(static_cast<double>(cols) / rows));
            if (error < min_error) {
                best = {rows, cols};
                min_error = error;
            } else if (error == min_error && rows > best[0]) {
                best = {rows, cols};
            }
        }
    }
    return best;
}

std::array<int32_t, 2> best_resize(double height, double width, int32_t scale_resolution,
                                   int32_t patch_size, int32_t merge_factor, bool allow_upscale) {
    const double scale = static_cast<double>(scale_resolution);
    if (height * width > scale * scale || allow_upscale) {
        const double aspect = width / height;
        const auto resized_height = static_cast<int32_t>(scale / std::sqrt(aspect));
        height = resized_height;
        width = static_cast<int32_t>(resized_height * aspect);
    }
    const int32_t divisor = patch_size * merge_factor;
    return {ensure_divide(height, divisor), ensure_divide(width, divisor)};
}

std::array<int32_t, 2> refine_size(int32_t height, int32_t width, std::array<int32_t, 2> grid,
                                   int32_t scale_resolution, int32_t patch_size,
                                   int32_t merge_factor) {
    const int32_t grid_y = grid[0], grid_x = grid[1];
    const int32_t refine_width = ensure_divide(width, grid_x);
    const int32_t refine_height = ensure_divide(height, grid_y);
    const auto best = best_resize(static_cast<double>(refine_height) / grid_y,
                                  static_cast<double>(refine_width) / grid_x, scale_resolution,
                                  patch_size, merge_factor, true);
    return {best[0] * grid_y, best[1] * grid_x};
}

std::vector<uint8_t> resize_bicubic_u8(const uint8_t* rgb, int32_t width, int32_t height,
                                       int32_t target_width, int32_t target_height) {
    if (rgb == nullptr || width <= 0 || height <= 0 || target_width <= 0 || target_height <= 0)
        throw std::invalid_argument("MiniCPM-V 4.7 resize requires positive dimensions");
    if (width == target_width && height == target_height)
        return {rgb, rgb + static_cast<std::size_t>(width) * height * 3};
    std::vector<uint8_t> horizontal;
    const uint8_t* source = rgb;
    if (width != target_width) {
        horizontal = resize_horizontal(rgb, width, height, target_width);
        source = horizontal.data();
    }
    if (height == target_height)
        return horizontal;
    return resize_vertical(source, target_width, height, target_height);
}

// Ports get_vision_nearest_position_ids (Transformers 5.22.2 vision_utils.py, Apache-2.0).
std::vector<int32_t> nearest_position_ids(int32_t grid_h, int32_t grid_w, int32_t per_side) {
    // The provider computes these on the model's GPU in FP32: torch.arange(1/per_side, 1,
    // 1/per_side) as start + step * i contracted to one FMA, torch.arange(size) / size as a
    // multiplication by the FP32 reciprocal, then bucketize(..., right=True).
    const auto step = static_cast<float>(1.0 / per_side);
    const auto count = static_cast<int32_t>(std::ceil((1.0 - 1.0 / per_side) / (1.0 / per_side)));
    std::vector<float> boundaries(static_cast<std::size_t>(count));
    for (int32_t i = 0; i < count; ++i)
        boundaries[static_cast<std::size_t>(i)] = std::fma(step, static_cast<float>(i), step);
    auto bucket = [&](int32_t index, int32_t size) {
        const float value = static_cast<float>(index) * (1.0F / static_cast<float>(size));
        return static_cast<int32_t>(std::upper_bound(boundaries.begin(), boundaries.end(), value) -
                                    boundaries.begin());
    };
    std::vector<int32_t> ids(static_cast<std::size_t>(grid_h) * grid_w);
    for (int32_t y = 0; y < grid_h; ++y) {
        const int32_t row = bucket(y, grid_h) * per_side;
        for (int32_t x = 0; x < grid_w; ++x)
            ids[static_cast<std::size_t>(y) * grid_w + x] = row + bucket(x, grid_w);
    }
    return ids;
}

// The 2x2 window order of get_vision_window_index (Transformers 5.22.2 vision_utils.py).
std::vector<int32_t> block_order(int32_t grid_h, int32_t grid_w) {
    if (grid_h % 2 || grid_w % 2)
        throw std::invalid_argument("MiniCPM-V 4.7 patch grids must split into 2x2 blocks");
    std::vector<int32_t> order;
    order.reserve(static_cast<std::size_t>(grid_h) * grid_w);
    for (int32_t by = 0; by < grid_h / 2; ++by)
        for (int32_t bx = 0; bx < grid_w / 2; ++bx)
            for (int32_t dy = 0; dy < 2; ++dy)
                for (int32_t dx = 0; dx < 2; ++dx)
                    order.push_back((2 * by + dy) * grid_w + 2 * bx + dx);
    return order;
}

ImageCrops preprocess_image(const uint8_t* rgb, int32_t height, int32_t width,
                            const ProcessorConfig& config) {
    if (rgb == nullptr || height <= 0 || width <= 0)
        throw std::invalid_argument("MiniCPM-V 4.7 image requires positive dimensions");
    ImageCrops result;
    const auto grid = sliced_grid(height, width, config.max_slice_nums, config.scale_resolution);
    const bool sliced = grid[0] > 0;
    const auto source = best_resize(height, width, config.scale_resolution, config.patch_size,
                                    config.merge_factor, !sliced);
    const auto thumbnail = resize_bicubic_u8(rgb, width, height, source[1], source[0]);
    result.crops.push_back(make_crop(thumbnail, source[1], 0, 0, source[0], source[1], config));
    if (!sliced)
        return result;
    result.rows = grid[0];
    result.cols = grid[1];
    const auto refine = refine_size(height, width, grid, config.scale_resolution, config.patch_size,
                                    config.merge_factor);
    const auto refined = resize_bicubic_u8(rgb, width, height, refine[1], refine[0]);
    const int32_t slice_h = refine[0] / grid[0], slice_w = refine[1] / grid[1];
    for (int32_t r = 0; r < grid[0]; ++r)
        for (int32_t c = 0; c < grid[1]; ++c)
            result.crops.push_back(
                make_crop(refined, refine[1], r * slice_h, c * slice_w, slice_h, slice_w, config));
    return result;
}

} // namespace trtmc::minicpmv4_7
