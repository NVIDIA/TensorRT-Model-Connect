/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Phi-4-multimodal Dynamic-HD preprocessing, reproducing the checkpoint's
// Phi4MMImageProcessor: the crop grid and padding of dynamic_preprocess, the
// per-crop patch masks, Pillow's bilinear resize of the image, the bicubic
// global view, and the sub_glb token layout of Phi4MMImageEmbedding.
//
// Ported from the checkpoint's processing_phi4mm.py and modeling_phi4mm.py
// (Apache-2.0), Pillow 12.2.0 src/libImaging/Resample.c (MIT-CMU) and PyTorch v2.12.0
// UpSample.h / cpu/UpSampleKernel.cpp (BSD-3-Clause); the notices are
// reproduced in families/phi4_multimodal/NOTICE.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace trtmc {
namespace phi4_multimodal {

inline constexpr int32_t kHdCropSize = 448;
inline constexpr int32_t kHdPatchSize = 14;
inline constexpr int32_t kHdPatchGrid = kHdCropSize / kHdPatchSize; // 32
inline constexpr int32_t kHdPooledGrid = kHdPatchGrid / 2;          // 16
inline constexpr int32_t kHdMaxCrops = 36;
// Gather rows the vision engine prepends to its pooled crop features.
inline constexpr int32_t kHdSubSeparatorRow = 0;
inline constexpr int32_t kHdGlobalSeparatorRow = 1;
inline constexpr int32_t kHdFirstFeatureRow = 2;

struct HdLayout {
    int32_t crop_cols{0};     // w crops
    int32_t crop_rows{0};     // h crops
    int32_t resized_width{0}; // image size before the white right/bottom padding
    int32_t resized_height{0};
    // Valid patch rows/cols of every vision batch entry: the global view first,
    // then the crops in row-major order.
    std::vector<int32_t> crop_valid;
    // Rows of [sub_GN, glb_GN, pooled features of every batch entry] that form
    // the image tokens, in the checkpoint's sub_glb order.
    std::vector<int32_t> token_rows;

    int32_t num_crops() const { return crop_cols * crop_rows; }
};

// dynamic_preprocess: the crop grid for an image (ties resolved as the
// checkpoint does, over target ratios ordered by crop count).
inline std::pair<int32_t, int32_t> hd_crop_grid(int32_t width, int32_t height,
                                                int32_t max_crops = kHdMaxCrops) {
    const auto cols = static_cast<int32_t>(std::ceil(width / static_cast<double>(kHdCropSize)));
    const auto rows = static_cast<int32_t>(std::ceil(height / static_cast<double>(kHdCropSize)));
    if (cols * rows <= max_crops)
        return {cols, rows};
    std::vector<std::pair<int32_t, int32_t>> ratios;
    for (int32_t count = 1; count <= max_crops; ++count)
        for (int32_t i = 1; i <= count; ++i)
            for (int32_t j = 1; j <= count; ++j)
                if (i * j == count)
                    ratios.emplace_back(i, j);
    const double aspect = static_cast<double>(width) / static_cast<double>(height);
    const double area = static_cast<double>(width) * static_cast<double>(height);
    double best_difference = std::numeric_limits<double>::infinity();
    std::pair<int32_t, int32_t> best{1, 1};
    for (const auto& ratio : ratios) {
        const double difference =
            std::abs(aspect - static_cast<double>(ratio.first) / static_cast<double>(ratio.second));
        if (difference < best_difference) {
            best_difference = difference;
            best = ratio;
        } else if (difference == best_difference &&
                   area > 0.5 * kHdCropSize * kHdCropSize * ratio.first * ratio.second) {
            best = ratio;
        }
    }
    return best;
}

inline HdLayout hd_layout(int32_t width, int32_t height, int32_t max_crops = kHdMaxCrops) {
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("Phi-4 Dynamic-HD: empty image");
    HdLayout layout;
    std::tie(layout.crop_cols, layout.crop_rows) = hd_crop_grid(width, height, max_crops);
    const int32_t target_width = kHdCropSize * layout.crop_cols;
    const int32_t target_height = kHdCropSize * layout.crop_rows;
    const double ratio_width = static_cast<double>(target_width) / static_cast<double>(width);
    const double ratio_height = static_cast<double>(target_height) / static_cast<double>(height);
    int32_t padding_width = 0;
    int32_t padding_height = 0;
    if (ratio_width < ratio_height) {
        layout.resized_width = target_width;
        layout.resized_height = static_cast<int32_t>(height * ratio_width);
        padding_height = target_height - layout.resized_height;
    } else {
        layout.resized_width = static_cast<int32_t>(width * ratio_height);
        layout.resized_height = target_height;
        padding_width = target_width - layout.resized_width;
    }
    if (std::min(layout.resized_height, target_height) < 10 ||
        std::min(layout.resized_width, target_width) < 10)
        throw std::invalid_argument("Phi-4 Dynamic-HD: the aspect ratio is very extreme");

    // The image attention mask zeroes whole padded patch columns and rows.
    const int32_t valid_cols = target_width / kHdPatchSize -
                               (padding_width >= kHdPatchSize ? padding_width / kHdPatchSize : 0);
    const int32_t valid_rows = target_height / kHdPatchSize -
                               (padding_height >= kHdPatchSize ? padding_height / kHdPatchSize : 0);
    layout.crop_valid = {kHdPatchGrid, kHdPatchGrid};
    for (int32_t row = 0; row < layout.crop_rows; ++row) {
        for (int32_t col = 0; col < layout.crop_cols; ++col) {
            const int32_t crop_valid_cols =
                std::clamp(valid_cols - col * kHdPatchGrid, 0, kHdPatchGrid);
            const int32_t crop_valid_rows =
                std::clamp(valid_rows - row * kHdPatchGrid, 0, kHdPatchGrid);
            // SiglipVisionEmbeddings counts rows on the crop's first column and
            // columns on its first row: a crop without valid columns has none.
            layout.crop_valid.push_back(crop_valid_cols > 0 ? crop_valid_rows : 0);
            layout.crop_valid.push_back(crop_valid_rows > 0 ? crop_valid_cols : 0);
        }
    }

    // sub_glb: the cropped sub-image grid with a separator after every row,
    // the global separator, then the global view with row separators. The
    // useful grid is the even-sampled mask (2x2 pooling keeps cell 0).
    const int32_t useful_rows = (valid_rows + 1) / 2;
    const int32_t useful_cols = (valid_cols + 1) / 2;
    const auto feature_row = [](int32_t entry, int32_t y, int32_t x) {
        return kHdFirstFeatureRow + entry * kHdPooledGrid * kHdPooledGrid + y * kHdPooledGrid + x;
    };
    layout.token_rows.reserve(static_cast<std::size_t>(useful_rows) * (useful_cols + 1) + 1 +
                              kHdPooledGrid * (kHdPooledGrid + 1));
    for (int32_t y = 0; y < useful_rows; ++y) {
        for (int32_t x = 0; x < useful_cols; ++x) {
            const int32_t crop = 1 + (y / kHdPooledGrid) * layout.crop_cols + x / kHdPooledGrid;
            layout.token_rows.push_back(feature_row(crop, y % kHdPooledGrid, x % kHdPooledGrid));
        }
        layout.token_rows.push_back(kHdSubSeparatorRow);
    }
    layout.token_rows.push_back(kHdGlobalSeparatorRow);
    for (int32_t y = 0; y < kHdPooledGrid; ++y) {
        for (int32_t x = 0; x < kHdPooledGrid; ++x)
            layout.token_rows.push_back(feature_row(0, y, x));
        layout.token_rows.push_back(kHdSubSeparatorRow);
    }
    return layout;
}

// Image.resize((width, height), Image.BILINEAR) on an 8-bit RGB image: Pillow's
// separable 22-bit fixed-point triangle filter, horizontal pass first, each
// pass rounded to 8 bits. Adapted from Pillow's Resample.c (precompute_coeffs,
// normalize_coeffs_8bpc, ImagingResampleHorizontal/Vertical_8bpc).
inline std::vector<uint8_t> pillow_bilinear_resize_rgb(const std::vector<uint8_t>& src,
                                                       int32_t height, int32_t width,
                                                       int32_t out_height, int32_t out_width) {
    if (height <= 0 || width <= 0 || out_height <= 0 || out_width <= 0 ||
        src.size() != static_cast<std::size_t>(height) * width * 3U)
        throw std::invalid_argument("pillow_bilinear_resize_rgb: invalid image geometry");
    constexpr int32_t precision = 22;
    struct Taps {
        int32_t first{0};
        std::vector<int32_t> weights;
    };
    const auto plan = [](int32_t in_size, int32_t out_size) {
        const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
        const double filter_scale = std::max(scale, 1.0);
        const double support = filter_scale;
        std::vector<Taps> rows(static_cast<std::size_t>(out_size));
        for (int32_t index = 0; index < out_size; ++index) {
            const double center = (static_cast<double>(index) + 0.5) * scale;
            const auto first = std::max(static_cast<int32_t>(center - support + 0.5), 0);
            const auto count =
                std::min(static_cast<int32_t>(center + support + 0.5), in_size) - first;
            std::vector<double> weights(static_cast<std::size_t>(std::max(count, 0)));
            double total = 0.0;
            for (int32_t tap = 0; tap < count; ++tap) {
                const double x =
                    std::abs((static_cast<double>(tap + first) - center + 0.5) / filter_scale);
                weights[static_cast<std::size_t>(tap)] = x < 1.0 ? 1.0 - x : 0.0;
                total += weights[static_cast<std::size_t>(tap)];
            }
            Taps taps{first, {}};
            for (double weight : weights) {
                if (total != 0.0)
                    weight /= total;
                const double scaled = weight * static_cast<double>(1 << precision);
                taps.weights.push_back(
                    static_cast<int32_t>(weight < 0.0 ? -0.5 + scaled : 0.5 + scaled));
            }
            rows[static_cast<std::size_t>(index)] = std::move(taps);
        }
        return rows;
    };
    const auto pass = [](const std::vector<uint8_t>& in, int32_t in_h, int32_t in_w,
                         const std::vector<Taps>& rows, bool along_width) {
        const auto out_h = along_width ? in_h : static_cast<int32_t>(rows.size());
        const auto out_w = along_width ? static_cast<int32_t>(rows.size()) : in_w;
        std::vector<uint8_t> out(static_cast<std::size_t>(out_h) * out_w * 3U);
        constexpr int64_t half = int64_t{1} << (precision - 1);
        for (int32_t y = 0; y < out_h; ++y) {
            for (int32_t x = 0; x < out_w; ++x) {
                const auto& taps = rows[static_cast<std::size_t>(along_width ? x : y)];
                for (int32_t c = 0; c < 3; ++c) {
                    int64_t sum = half;
                    for (std::size_t tap = 0; tap < taps.weights.size(); ++tap) {
                        const auto source = static_cast<int32_t>(tap) + taps.first;
                        const auto sy = along_width ? y : source;
                        const auto sx = along_width ? source : x;
                        sum += static_cast<int64_t>(
                                   in[(static_cast<std::size_t>(sy) * in_w + sx) * 3U + c]) *
                               taps.weights[tap];
                    }
                    out[(static_cast<std::size_t>(y) * out_w + x) * 3U + c] =
                        static_cast<uint8_t>(std::clamp<int64_t>(sum >> precision, 0, 255));
                }
            }
        }
        return out;
    };
    auto image = src;
    if (out_width != width)
        image = pass(image, height, width, plan(width, out_width), true);
    if (out_height != height)
        image = pass(image, height, out_width, plan(height, out_height), false);
    return image;
}

// F.interpolate(x, size, mode="bicubic", align_corners=False) on a float CHW
// image on CPU (no antialiasing, A = -0.75, clamped borders): the height
// weights wrap the width interpolation, accumulated in float. Adapted from
// PyTorch's cubic_convolution1/2 and CPU bicubic upsampling kernel.
inline std::vector<float> torch_bicubic_resize_chw(const std::vector<float>& src, int32_t channels,
                                                   int32_t height, int32_t width,
                                                   int32_t out_height, int32_t out_width) {
    if (channels <= 0 || height <= 0 || width <= 0 || out_height <= 0 || out_width <= 0 ||
        src.size() != static_cast<std::size_t>(channels) * height * width)
        throw std::invalid_argument("torch_bicubic_resize_chw: invalid image geometry");
    constexpr float a = -0.75F;
    const auto cubic1 = [](float x) { return ((a + 2.0F) * x - (a + 3.0F)) * x * x + 1.0F; };
    const auto cubic2 = [](float x) { return ((a * x - 5.0F * a) * x + 8.0F * a) * x - 4.0F * a; };
    struct Axis {
        std::vector<int32_t> index;
        std::vector<float> weight;
    };
    const auto axis = [&](int32_t in_size, int32_t out_size) {
        const float scale = static_cast<float>(in_size) / static_cast<float>(out_size);
        Axis result;
        for (int32_t out = 0; out < out_size; ++out) {
            const float source = scale * (static_cast<float>(out) + 0.5F) - 0.5F;
            const auto first = static_cast<int32_t>(std::floor(source));
            const float t = source - static_cast<float>(first);
            const float weights[4] = {cubic2(t + 1.0F), cubic1(t), cubic1(1.0F - t),
                                      cubic2(2.0F - t)};
            for (int32_t tap = 0; tap < 4; ++tap) {
                result.index.push_back(std::clamp(first - 1 + tap, 0, in_size - 1));
                result.weight.push_back(weights[tap]);
            }
        }
        return result;
    };
    const Axis rows = axis(height, out_height);
    const Axis cols = axis(width, out_width);
    std::vector<float> out(static_cast<std::size_t>(channels) * out_height * out_width);
    for (int32_t c = 0; c < channels; ++c) {
        const float* plane = src.data() + static_cast<std::size_t>(c) * height * width;
        for (int32_t y = 0; y < out_height; ++y) {
            for (int32_t x = 0; x < out_width; ++x) {
                float value = 0.0F;
                for (int32_t i = 0; i < 4; ++i) {
                    const float* line =
                        plane +
                        static_cast<std::size_t>(rows.index[static_cast<std::size_t>(y * 4 + i)]) *
                            width;
                    float row = 0.0F;
                    for (int32_t j = 0; j < 4; ++j) {
                        const auto k = static_cast<std::size_t>(x * 4 + j);
                        const float term = line[cols.index[k]] * cols.weight[k];
                        row = j == 0 ? term : row + term;
                    }
                    const float term = row * rows.weight[static_cast<std::size_t>(y * 4 + i)];
                    value = i == 0 ? term : value + term;
                }
                out[(static_cast<std::size_t>(c) * out_height + y) * out_width + x] = value;
            }
        }
    }
    return out;
}

} // namespace phi4_multimodal
} // namespace trtmc
