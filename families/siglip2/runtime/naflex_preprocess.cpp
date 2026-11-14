/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapted from transformers v5.2.0 (patch-budget size search, patch layout), Pillow 13.0.0
// src/libImaging/Resample.c (8-bit bilinear resize) and PyTorch v2.12.0
// aten/src/ATen/native/cpu/UpSampleKernel.cpp (antialiased taps); see ../NOTICE.

#include "families/siglip2/runtime/naflex_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

namespace {

constexpr std::int32_t kChannels = 3;

// ---- Target size: transformers get_image_size_for_max_num_patches -----------------

// One side scaled, rounded up to whole patches, never below one patch. The reference
// computes this in Python floats; every operation here is the same IEEE double one.
std::int64_t scaled_side(double scale, std::int32_t size, std::int32_t patch) {
    const double scaled = static_cast<double>(size) * scale;
    const auto patches = static_cast<std::int64_t>(std::ceil(scaled / static_cast<double>(patch)));
    return std::max<std::int64_t>(patch, patches * patch);
}

// ---- Pillow 13.0.0 bilinear resize (Resample.c) -------------------------------------

// Pillow's Resample.c: 8-bit passes use 22 fractional bits and round half up.
constexpr std::int32_t kPrecisionBits = 32 - 8 - 2;

struct ResampleAxis {
    std::int32_t taps{0}; // stride of `weights` per output index
    std::vector<std::int32_t> first;
    std::vector<std::int32_t> count;
    std::vector<std::int32_t> weights;
};

double pillow_bilinear(double x) {
    if (x < 0.0)
        x = -x;
    if (x < 1.0)
        return 1.0 - x;
    return 0.0;
}

// precompute_coeffs + normalize_coeffs_8bpc for the default box (0, size).
ResampleAxis pillow_axis(std::int32_t in_size, std::int32_t out_size) {
    const double scale = static_cast<double>(static_cast<float>(in_size)) / out_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 1.0 * filter_scale;
    ResampleAxis axis;
    axis.taps = static_cast<std::int32_t>(std::ceil(support)) * 2 + 1;
    axis.first.resize(static_cast<std::size_t>(out_size));
    axis.count.resize(static_cast<std::size_t>(out_size));
    axis.weights.assign(static_cast<std::size_t>(out_size) * axis.taps, 0);
    std::vector<double> floating(static_cast<std::size_t>(axis.taps));
    for (std::int32_t out = 0; out < out_size; ++out) {
        const double center = 0.0 + (out + 0.5) * scale;
        const double inverse = 1.0 / filter_scale;
        auto first = static_cast<std::int32_t>(center - support + 0.5);
        first = std::max(first, 0);
        auto last = static_cast<std::int32_t>(center + support + 0.5);
        last = std::min(last, in_size);
        const std::int32_t count = std::min(last - first, axis.taps);
        double total = 0.0;
        for (std::int32_t tap = 0; tap < count; ++tap) {
            const double weight = pillow_bilinear((tap + first - center + 0.5) * inverse);
            floating[static_cast<std::size_t>(tap)] = weight;
            total += weight;
        }
        for (std::int32_t tap = 0; tap < count; ++tap) {
            double weight = floating[static_cast<std::size_t>(tap)];
            if (total != 0.0)
                weight /= total;
            const double scaled = weight * (1 << kPrecisionBits);
            axis.weights[static_cast<std::size_t>(out) * axis.taps + tap] =
                static_cast<std::int32_t>(scaled < 0 ? -0.5 + scaled : 0.5 + scaled);
        }
        axis.first[static_cast<std::size_t>(out)] = first;
        axis.count[static_cast<std::size_t>(out)] = std::max(count, 0);
    }
    return axis;
}

std::uint8_t pillow_clip(std::int64_t sum) {
    const std::int64_t value = sum >> kPrecisionBits;
    return static_cast<std::uint8_t>(std::clamp<std::int64_t>(value, 0, 255));
}

// One output value of a pass: `stride` is the step between consecutive source taps.
std::uint8_t pillow_tap_sum(const std::uint8_t* source, std::size_t stride,
                            const ResampleAxis& axis, std::int32_t out) {
    std::int64_t sum = std::int64_t{1} << (kPrecisionBits - 1);
    const auto base = static_cast<std::size_t>(out) * axis.taps;
    const auto first = static_cast<std::size_t>(axis.first[static_cast<std::size_t>(out)]);
    for (std::int32_t tap = 0; tap < axis.count[static_cast<std::size_t>(out)]; ++tap) {
        sum += static_cast<std::int64_t>(source[(first + static_cast<std::size_t>(tap)) * stride]) *
               axis.weights[base + static_cast<std::size_t>(tap)];
    }
    return pillow_clip(sum);
}

std::vector<std::uint8_t> resize_horizontal(const std::vector<std::uint8_t>& image,
                                            std::int32_t height, std::int32_t width,
                                            std::int32_t out_width) {
    const auto axis = pillow_axis(width, out_width);
    std::vector<std::uint8_t> output(static_cast<std::size_t>(height) * out_width * kChannels);
    for (std::int32_t y = 0; y < height; ++y) {
        const auto* row = image.data() + static_cast<std::size_t>(y) * width * kChannels;
        for (std::int32_t x = 0; x < out_width; ++x) {
            for (std::int32_t c = 0; c < kChannels; ++c) {
                output[(static_cast<std::size_t>(y) * out_width + x) * kChannels + c] =
                    pillow_tap_sum(row + c, kChannels, axis, x);
            }
        }
    }
    return output;
}

std::vector<std::uint8_t> resize_vertical(const std::vector<std::uint8_t>& image,
                                          std::int32_t height, std::int32_t width,
                                          std::int32_t out_height) {
    const auto axis = pillow_axis(height, out_height);
    const auto row_stride = static_cast<std::size_t>(width) * kChannels;
    std::vector<std::uint8_t> output(static_cast<std::size_t>(out_height) * row_stride);
    for (std::int32_t y = 0; y < out_height; ++y) {
        for (std::size_t column = 0; column < row_stride; ++column) {
            output[static_cast<std::size_t>(y) * row_stride + column] =
                pillow_tap_sum(image.data() + column, row_stride, axis, y);
        }
    }
    return output;
}

// ---- Position-table taps: PyTorch _compute_indices_min_size_weights_aa -------------

double torch_bilinear(double x) {
    x = std::abs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}

void require_positive(std::int32_t value, const char* what) {
    if (value <= 0)
        throw std::invalid_argument(std::string("siglip2 ") + what + " must be positive");
}

void validate(const Siglip2PreprocessConfig& config) {
    require_positive(config.patch_size, "patch size");
    require_positive(config.max_num_patches, "patch budget");
    require_positive(config.position_grid, "position grid");
    if (config.image_mean.size() != kChannels || config.image_std.size() != kChannels)
        throw std::invalid_argument("siglip2 image mean/std must have three channels");
    for (const float value : config.image_std) {
        if (value == 0.0F)
            throw std::invalid_argument("siglip2 image std must be non-zero");
    }
    if (!(config.rescale_factor > 0.0))
        throw std::invalid_argument("siglip2 rescale factor must be positive");
}

std::vector<std::uint8_t> to_bytes(const float* rgb, std::int32_t height, std::int32_t width) {
    std::vector<std::uint8_t> image(static_cast<std::size_t>(height) * width * kChannels);
    for (std::size_t index = 0; index < image.size(); ++index) {
        const float value = std::clamp(rgb[index], 0.0F, 1.0F);
        image[index] = static_cast<std::uint8_t>(std::lround(value * 255.0F));
    }
    return image;
}

std::vector<float> patchify(const std::vector<std::uint8_t>& image, std::int32_t grid_height,
                            std::int32_t grid_width, const Siglip2PreprocessConfig& config) {
    const std::int32_t patch = config.patch_size;
    const std::int32_t width = grid_width * patch;
    const auto features = static_cast<std::size_t>(patch) * patch * kChannels;
    std::vector<float> values(static_cast<std::size_t>(grid_height) * grid_width * features);
    std::size_t out = 0;
    for (std::int32_t py = 0; py < grid_height; ++py) {
        for (std::int32_t px = 0; px < grid_width; ++px) {
            for (std::int32_t ry = 0; ry < patch; ++ry) {
                const auto row = static_cast<std::size_t>(py * patch + ry) * width;
                for (std::int32_t rx = 0; rx < patch; ++rx) {
                    const auto pixel =
                        (row + static_cast<std::size_t>(px * patch + rx)) * kChannels;
                    for (std::int32_t c = 0; c < kChannels; ++c) {
                        // rescale in double then store float32, as numpy does; normalise
                        // in float32.
                        const auto scaled = static_cast<float>(
                            static_cast<double>(image[pixel + static_cast<std::size_t>(c)]) *
                            config.rescale_factor);
                        values[out++] = (scaled - config.image_mean[static_cast<std::size_t>(c)]) /
                                        config.image_std[static_cast<std::size_t>(c)];
                    }
                }
            }
        }
    }
    return values;
}

std::vector<float> position_taps(std::int32_t grid_height, std::int32_t grid_width,
                                 std::int32_t table_grid) {
    const auto rows = siglip2_axis_taps(table_grid, grid_height);
    const auto columns = siglip2_axis_taps(table_grid, grid_width);
    const auto table = static_cast<std::size_t>(table_grid) * table_grid;
    std::vector<float> taps(static_cast<std::size_t>(grid_height) * grid_width * table, 0.0F);
    for (std::int32_t y = 0; y < grid_height; ++y) {
        for (std::int32_t x = 0; x < grid_width; ++x) {
            float* out = taps.data() + (static_cast<std::size_t>(y) * grid_width + x) * table;
            for (std::int32_t sy = 0; sy < table_grid; ++sy) {
                const double wy = rows[static_cast<std::size_t>(y) * table_grid + sy];
                if (wy == 0.0)
                    continue;
                for (std::int32_t sx = 0; sx < table_grid; ++sx) {
                    const double wx = columns[static_cast<std::size_t>(x) * table_grid + sx];
                    out[static_cast<std::size_t>(sy) * table_grid + sx] =
                        static_cast<float>(wy * wx);
                }
            }
        }
    }
    return taps;
}

} // namespace

std::pair<std::int32_t, std::int32_t> siglip2_target_size(std::int32_t height, std::int32_t width,
                                                          std::int32_t patch_size,
                                                          std::int32_t max_num_patches) {
    if (height <= 0 || width <= 0 || patch_size <= 0 || max_num_patches <= 0)
        throw std::invalid_argument("siglip2 target size needs positive dimensions");
    constexpr double kEps = 1e-5;
    double scale_min = kEps / 10;
    double scale_max = 100.0;
    while ((scale_max - scale_min) >= kEps) {
        const double scale = (scale_min + scale_max) / 2;
        const auto rows = scaled_side(scale, height, patch_size) / patch_size;
        const auto columns = scaled_side(scale, width, patch_size) / patch_size;
        if (static_cast<double>(rows) * static_cast<double>(columns) <= max_num_patches)
            scale_min = scale;
        else
            scale_max = scale;
    }
    const auto target_height = scaled_side(scale_min, height, patch_size);
    const auto target_width = scaled_side(scale_min, width, patch_size);
    if (target_height > std::numeric_limits<std::int32_t>::max() ||
        target_width > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("siglip2 image is too large to resize");
    }
    return {static_cast<std::int32_t>(target_height), static_cast<std::int32_t>(target_width)};
}

std::vector<std::uint8_t> siglip2_resize_bilinear(const std::uint8_t* rgb, std::int32_t height,
                                                  std::int32_t width, std::int32_t out_height,
                                                  std::int32_t out_width) {
    if (rgb == nullptr || height <= 0 || width <= 0 || out_height <= 0 || out_width <= 0)
        throw std::invalid_argument("siglip2 resize needs a non-empty image");
    std::vector<std::uint8_t> image(rgb,
                                    rgb + static_cast<std::size_t>(height) * width * kChannels);
    // Pillow skips an axis whose size does not change. Each pass rounds to 8 bits, so the
    // order matters: Pillow 13 resamples vertically first when the height shrinks by more
    // than twice as many pixels as the width (Pillow 12 always went horizontally first).
    const std::int64_t shrink_height = static_cast<std::int64_t>(height) - out_height;
    const std::int64_t shrink_width = static_cast<std::int64_t>(width) - out_width;
    const bool horizontal_first = !(shrink_height > 0 && shrink_height > shrink_width * 2);
    if (horizontal_first) {
        if (out_width != width)
            image = resize_horizontal(image, height, width, out_width);
        if (out_height != height)
            image = resize_vertical(image, height, out_width, out_height);
    } else {
        if (out_height != height)
            image = resize_vertical(image, height, width, out_height);
        if (out_width != width)
            image = resize_horizontal(image, out_height, width, out_width);
    }
    return image;
}

std::vector<double> siglip2_axis_taps(std::int32_t input, std::int32_t output) {
    if (input <= 0 || output <= 0)
        throw std::invalid_argument("siglip2 position taps need positive sizes");
    const double scale = static_cast<double>(input) / output;
    const double support = scale >= 1.0 ? scale : 1.0;
    const double inverse = scale >= 1.0 ? 1.0 / scale : 1.0;
    std::vector<double> taps(static_cast<std::size_t>(output) * input, 0.0);
    for (std::int32_t out = 0; out < output; ++out) {
        const double center = scale * (out + 0.5);
        const auto first =
            std::max<std::int64_t>(static_cast<std::int64_t>(center - support + 0.5), 0);
        const auto last =
            std::min<std::int64_t>(static_cast<std::int64_t>(center + support + 0.5), input);
        double total = 0.0;
        double* row = taps.data() + static_cast<std::size_t>(out) * input;
        for (std::int64_t index = first; index < last; ++index) {
            const double weight =
                torch_bilinear((static_cast<double>(index) - center + 0.5) * inverse);
            row[index] = weight;
            total += weight;
        }
        if (total != 0.0) {
            for (std::int64_t index = first; index < last; ++index)
                row[index] /= total;
        }
    }
    return taps;
}

Siglip2Patches preprocess_siglip2_image(const float* rgb, std::int32_t height, std::int32_t width,
                                        const Siglip2PreprocessConfig& config) {
    if (rgb == nullptr || height <= 0 || width <= 0)
        throw std::invalid_argument("siglip2 source image must be non-empty");
    validate(config);

    const auto [target_height, target_width] =
        siglip2_target_size(height, width, config.patch_size, config.max_num_patches);
    const auto source = to_bytes(rgb, height, width);
    const auto resized =
        siglip2_resize_bilinear(source.data(), height, width, target_height, target_width);

    Siglip2Patches patches;
    patches.grid_height = target_height / config.patch_size;
    patches.grid_width = target_width / config.patch_size;
    if (static_cast<std::int64_t>(patches.grid_height) * patches.grid_width >
        config.max_num_patches)
        throw std::runtime_error("siglip2 resize exceeded the patch budget");
    patches.pixel_values = patchify(resized, patches.grid_height, patches.grid_width, config);
    patches.position_weights =
        position_taps(patches.grid_height, patches.grid_width, config.position_grid);
    return patches;
}

} // namespace trtmc
