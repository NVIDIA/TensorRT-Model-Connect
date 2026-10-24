/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_vl/runtime/image_processor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::lfm2_vl {
namespace {
struct Coefficients {
    int start;
    std::vector<double> weights;
};
std::vector<Coefficients> coefficients(int input, int output) {
    const double scale = static_cast<double>(input) / output;
    const double support = std::max(scale, 1.0);
    std::vector<Coefficients> result;
    for (int index = 0; index < output; ++index) {
        const double center = scale * (index + 0.5);
        const int first = std::max(0, static_cast<int>(center - support + 0.5));
        const int end = std::min(input, static_cast<int>(center + support + 0.5));
        Coefficients c{first, {}};
        double total = 0;
        for (int j = first; j < end; ++j) {
            const double weight = std::max(0.0, 1.0 - std::abs((j + 0.5 - center) / support));
            c.weights.push_back(weight);
            total += weight;
        }
        for (auto& w : c.weights)
            w /= total;
        result.push_back(std::move(c));
    }
    return result;
}
int round_factor(int value) {
    return static_cast<int>(std::nearbyint(value / 32.0)) * 32;
}
std::pair<int, int> smart_size(int h, int w, const ImageOptions& opt) {
    int oh = std::max(32, round_factor(h)), ow = std::max(32, round_factor(w));
    const double area = static_cast<double>(h) * w;
    if (static_cast<double>(oh) * ow > opt.max_tokens * 1024.0) {
        const double beta = std::sqrt(area / (opt.max_tokens * 1024.0));
        oh = std::max(32, static_cast<int>(std::floor(h / beta / 32)) * 32);
        ow = std::max(32, static_cast<int>(std::floor(w / beta / 32)) * 32);
    } else if (static_cast<double>(oh) * ow < opt.min_tokens * 1024.0) {
        const double beta = std::sqrt(opt.min_tokens * 1024.0 / area);
        oh = static_cast<int>(std::ceil(h * beta / 32)) * 32;
        ow = static_cast<int>(std::ceil(w * beta / 32)) * 32;
    }
    return {oh, ow};
}
std::pair<int, int> tile_grid(int h, int w, int max_tiles) {
    std::vector<std::pair<int, int>> ratios;
    for (int cols = 1; cols <= max_tiles; ++cols)
        for (int rows = 1; rows <= max_tiles; ++rows)
            if (cols * rows >= 2 && cols * rows <= max_tiles)
                ratios.emplace_back(cols, rows);
    std::stable_sort(ratios.begin(), ratios.end(),
                     [](auto a, auto b) { return a.first * a.second < b.first * b.second; });
    double best = std::numeric_limits<double>::infinity();
    std::pair<int, int> selected{1, 1};
    for (auto ratio : ratios) {
        const double diff =
            std::abs(static_cast<double>(w) / h - static_cast<double>(ratio.first) / ratio.second);
        if (diff < best || (diff == best && static_cast<double>(w) * h >
                                                0.5 * 512 * 512 * ratio.first * ratio.second)) {
            best = diff;
            selected = ratio;
        }
    }
    return selected;
}
ImageTile patchify(const std::vector<float>& rgb, int h, int w) {
    ImageTile tile;
    tile.patch_rows = h / 16;
    tile.patch_cols = w / 16;
    tile.pixels.reserve(static_cast<std::size_t>(h) * w * 3);
    for (int py = 0; py < h; py += 16)
        for (int px = 0; px < w; px += 16)
            for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 16; ++x)
                    for (int c = 0; c < 3; ++c)
                        tile.pixels.push_back((rgb[((py + y) * w + px + x) * 3 + c] - 127.5F) /
                                              127.5F);
    for (int y = 0; y < tile.patch_rows; y += 2)
        for (int x = 0; x < tile.patch_cols; x += 2)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                    tile.unshuffle.push_back((y + dy) * tile.patch_cols + x + dx);
    return tile;
}
void image_tokens(std::string& out, int count) {
    for (int i = 0; i < count; ++i)
        out += "<image>";
}
} // namespace

std::vector<float> resize_bilinear(const std::vector<float>& pixels, int height, int width,
                                   int channels, int out_height, int out_width, bool quantize) {
    if (height == out_height && width == out_width)
        return pixels;
    const auto horizontal = coefficients(width, out_width),
               vertical = coefficients(height, out_height);
    auto precision = [](const std::vector<Coefficients>& axis) {
        double maximum = 0;
        for (const auto& c : axis)
            for (double weight : c.weights)
                maximum = std::max(maximum, weight);
        int bits = 0;
        while (bits < 22 && static_cast<int>(0.5 + maximum * (1 << (bits + 1))) < (1 << 15))
            ++bits;
        return bits;
    };
    const int xbits = precision(horizontal), ybits = precision(vertical);
    std::vector<float> tmp(static_cast<std::size_t>(height) * out_width * channels);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < out_width; ++x)
            for (int c = 0; c < channels; ++c) {
                float value = 0;
                const auto& coeff = horizontal[x];
                if (quantize) {
                    std::int64_t sum = std::int64_t{1} << (xbits - 1);
                    for (std::size_t j = 0; j < coeff.weights.size(); ++j)
                        sum += static_cast<int>(
                                   pixels[(static_cast<std::size_t>(y) * width + coeff.start + j) *
                                              channels +
                                          c]) *
                               static_cast<int>(0.5 + coeff.weights[j] * (1 << xbits));
                    value = static_cast<float>(std::clamp<std::int64_t>(sum >> xbits, 0, 255));
                } else
                    for (std::size_t j = 0; j < coeff.weights.size(); ++j)
                        value += pixels[(static_cast<std::size_t>(y) * width + coeff.start + j) *
                                            channels +
                                        c] *
                                 static_cast<float>(coeff.weights[j]);
                tmp[(static_cast<std::size_t>(y) * out_width + x) * channels + c] = value;
            }
    std::vector<float> result(static_cast<std::size_t>(out_height) * out_width * channels);
    for (int y = 0; y < out_height; ++y)
        for (int x = 0; x < out_width; ++x)
            for (int c = 0; c < channels; ++c) {
                float value = 0;
                const auto& coeff = vertical[y];
                if (quantize) {
                    std::int64_t sum = std::int64_t{1} << (ybits - 1);
                    for (std::size_t j = 0; j < coeff.weights.size(); ++j)
                        sum += static_cast<int>(
                                   tmp[((coeff.start + j) * out_width + x) * channels + c]) *
                               static_cast<int>(0.5 + coeff.weights[j] * (1 << ybits));
                    value = static_cast<float>(std::clamp<std::int64_t>(sum >> ybits, 0, 255));
                } else
                    for (std::size_t j = 0; j < coeff.weights.size(); ++j)
                        value += tmp[((coeff.start + j) * out_width + x) * channels + c] *
                                 static_cast<float>(coeff.weights[j]);
                result[(static_cast<std::size_t>(y) * out_width + x) * channels + c] = value;
            }
    return result;
}

ProcessedImage process_image(const ImageResult& image, const ImageOptions& opt) {
    const int h = image.height, w = image.width;
    if (h <= 0 || w <= 0 || image.channels != 3 || image.num_frames != 1 ||
        image.pixels.size() != static_cast<std::size_t>(h) * w * 3)
        throw std::invalid_argument("LFM2-VL requires a nonempty RGB HWC image");
    if (opt.min_tokens < 1 || opt.max_tokens < opt.min_tokens || opt.max_tokens > 1024 ||
        opt.max_tiles < 1 || opt.max_tiles > 10 || (opt.split && opt.max_tiles < 2))
        throw std::invalid_argument("LFM2-VL image budget exceeds the bundle capacity");
    for (float value : image.pixels)
        if (!std::isfinite(value) || value < 0 || value > 255)
            throw std::invalid_argument("LFM2-VL image pixels must be finite and in [0,255]");
    const auto [oh, ow] = smart_size(h, w, opt);
    const bool large =
        static_cast<double>(std::max(16, round_factor(h))) * std::max(16, round_factor(w)) >
        opt.max_tokens * 2048.0;
    auto [cols, rows] = large && opt.split && opt.max_tiles > 1 ? tile_grid(h, w, opt.max_tiles)
                                                                : std::pair<int, int>{1, 1};
    ProcessedImage output;
    output.placeholders = "<|image_start|>";
    if (cols * rows > 1) {
        auto tiled =
            resize_bilinear(image.pixels, h, w, 3, rows * 512, cols * 512, opt.integer_pixels);
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < cols; ++col) {
                std::vector<float> crop(512 * 512 * 3);
                for (int y = 0; y < 512; ++y)
                    std::copy_n(tiled.begin() + ((row * 512 + y) * cols * 512 + col * 512) * 3,
                                512 * 3, crop.begin() + y * 512 * 3);
                output.tiles.push_back(patchify(crop, 512, 512));
                output.placeholders += "<|img_row_" + std::to_string(row + 1) + "_col_" +
                                       std::to_string(col + 1) + "|>";
                image_tokens(output.placeholders, 256);
            }
        output.placeholders += "<|img_thumbnail|>";
    }
    output.tiles.push_back(
        patchify(resize_bilinear(image.pixels, h, w, 3, oh, ow, opt.integer_pixels), oh, ow));
    image_tokens(output.placeholders, oh * ow / 1024);
    output.placeholders += "<|image_end|>";
    return output;
}
namespace {
float position_round(float value, bool bfloat16) {
    if (bfloat16) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        bits += 0x7fffU + ((bits >> 16U) & 1U);
        bits &= 0xffff0000U;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    if (value == 0.0F)
        return value;
    int exponent;
    std::frexp(value, &exponent);
    const float quantum = std::ldexp(1.0F, std::max(exponent - 11, -24));
    return std::nearbyint(value / quantum) * quantum;
}
std::vector<Coefficients> position_coefficients(int input, int output, bool bfloat16) {
    const float scale = static_cast<float>(input) / output;
    const float support = std::max(scale, 1.0F);
    const float inverse = scale >= 1.0F ? 1.0F / scale : 1.0F;
    std::vector<Coefficients> result;
    for (int index = 0; index < output; ++index) {
        const float center = scale * (index + 0.5F);
        const int first = std::max(0, static_cast<int>(center - support + 0.5F));
        const int end = std::min(input, static_cast<int>(center + support + 0.5F));
        Coefficients c{first, {}};
        float total = 0.0F;
        for (int j = 0; j < end - first; ++j) {
            const float value =
                std::max(0.0F, 1.0F - std::abs((j + (first - center) + 0.5F) * inverse));
            c.weights.push_back(position_round(value, bfloat16));
            total += value;
        }
        for (auto& weight : c.weights)
            weight = position_round(static_cast<float>(weight) / total, bfloat16);
        result.push_back(std::move(c));
    }
    return result;
}
} // namespace
std::vector<float> position_embeddings(const std::vector<float>& table, int grid, int hidden,
                                       int rows, int cols, bool bfloat16) {
    if (table.size() != static_cast<std::size_t>(grid) * grid * hidden || rows <= 0 || cols <= 0)
        throw std::invalid_argument("LFM2-VL position embedding table shape is invalid");
    // SigLIP2 interpolates the learned table in the model dtype. Coefficients
    // and each separable-pass output are rounded before the following pass.
    const auto horizontal = position_coefficients(grid, cols, bfloat16);
    const auto vertical = position_coefficients(grid, rows, bfloat16);
    std::vector<float> intermediate(static_cast<std::size_t>(grid) * cols * hidden);
    for (int y = 0; y < grid; ++y)
        for (int x = 0; x < cols; ++x)
            for (int c = 0; c < hidden; ++c) {
                const auto& coeff = horizontal[x];
                float sum = 0.0F;
                for (std::size_t j = 0; j < coeff.weights.size(); ++j)
                    sum +=
                        position_round(
                            table[(static_cast<std::size_t>(y) * grid + coeff.start + j) * hidden +
                                  c],
                            bfloat16) *
                        static_cast<float>(coeff.weights[j]);
                intermediate[(static_cast<std::size_t>(y) * cols + x) * hidden + c] =
                    position_round(sum, bfloat16);
            }
    std::vector<float> result(static_cast<std::size_t>(rows) * cols * hidden);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            for (int c = 0; c < hidden; ++c) {
                const auto& coeff = vertical[y];
                float sum = 0.0F;
                for (std::size_t j = 0; j < coeff.weights.size(); ++j)
                    sum += intermediate[((coeff.start + j) * cols + x) * hidden + c] *
                           static_cast<float>(coeff.weights[j]);
                result[(static_cast<std::size_t>(y) * cols + x) * hidden + c] =
                    position_round(sum, bfloat16);
            }
    return result;
}
VisionInputs prepare_vision_inputs(const ImageTile& tile, const ImageOptions& options,
                                   const std::vector<float>& position_table, int hidden,
                                   bool bfloat16) {
    VisionInputs result;
    result.valid_patches = static_cast<std::int64_t>(tile.patch_rows) * tile.patch_cols;
    result.patch_count = std::max(options.max_tokens * 4, options.split ? 1024 : 0);
    if (result.valid_patches <= 0 || result.valid_patches > result.patch_count ||
        tile.pixels.size() != static_cast<std::size_t>(result.valid_patches) * 768)
        throw std::invalid_argument("LFM2-VL resized patches exceed the image budget");
    result.pixels = tile.pixels;
    result.pixels.resize(static_cast<std::size_t>(result.patch_count) * 768, 0.0F);
    result.mask.assign(static_cast<std::size_t>(result.patch_count), 0);
    std::fill_n(result.mask.begin(), static_cast<std::size_t>(result.valid_patches), 1);
    result.positions =
        position_embeddings(position_table, 16, hidden, tile.patch_rows, tile.patch_cols, bfloat16);
    result.positions.resize(static_cast<std::size_t>(result.patch_count) * hidden);
    for (std::int64_t row = result.valid_patches; row < result.patch_count; ++row)
        std::copy_n(result.positions.begin(), hidden, result.positions.begin() + row * hidden);
    return result;
}
} // namespace trtmc::lfm2_vl
