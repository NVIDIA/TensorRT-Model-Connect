/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Ports SmolVLMImageProcessor (Transformers 5.2.0, image_processing_smolvlm.py) and the
// Lanczos resampling of Pillow 13.0.0 (src/libImaging/Resample.c); see ../NOTICE.
#include "media.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace trtmc::smolvlm {
namespace {

constexpr int kPrecisionBits = 32 - 8 - 2;
constexpr double kLanczosSupport = 3.0;
constexpr int kMaxImageSize = 4096; // The processor's absolute bound on either side.
constexpr double kPi = 3.14159265358979323846;

double sinc(double x) {
    if (x == 0.0)
        return 1.0;
    x = x * kPi;
    return std::sin(x) / x;
}

double lanczos(double x) {
    if (-3.0 <= x && x < 3.0)
        return sinc(x) * sinc(x / 3);
    return 0.0;
}

struct Coefficients {
    int ksize{0};
    std::vector<int> bounds;    // Per output sample: first input index, tap count.
    std::vector<int32_t> fixed; // [out, ksize] fixed-point weights.
};

// Pillow precompute_coeffs + normalize_coeffs_8bpc for box [0, in_size).
Coefficients coefficients(int in_size, int out_size) {
    const float in0 = 0.0F, in1 = static_cast<float>(in_size);
    const double scale = static_cast<double>(in1 - in0) / out_size;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = kLanczosSupport * filterscale;
    Coefficients result;
    result.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    result.bounds.resize(static_cast<size_t>(out_size) * 2);
    result.fixed.assign(static_cast<size_t>(out_size) * result.ksize, 0);
    std::vector<double> k(static_cast<size_t>(result.ksize));
    const double inv_filterscale = 1.0 / filterscale;
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = in0 + (xx + 0.5) * scale;
        double ww = 0.0;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0)
            xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size)
            xmax = in_size;
        xmax -= xmin;
        std::fill(k.begin(), k.end(), 0.0);
        for (int x = 0; x < xmax; ++x) {
            const double w = lanczos((x + xmin - center + 0.5) * inv_filterscale);
            k[static_cast<size_t>(x)] = w;
            ww += w;
        }
        if (ww != 0.0)
            for (int x = 0; x < xmax; ++x)
                k[static_cast<size_t>(x)] /= ww;
        for (int x = 0; x < result.ksize; ++x) {
            const double v = k[static_cast<size_t>(x)] * (1 << kPrecisionBits);
            result.fixed[static_cast<size_t>(xx) * result.ksize + x] =
                static_cast<int32_t>(k[static_cast<size_t>(x)] < 0 ? -0.5 + v : 0.5 + v);
        }
        result.bounds[static_cast<size_t>(xx) * 2] = xmin;
        result.bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }
    return result;
}

uint8_t clip8(int value) {
    return static_cast<uint8_t>(std::clamp(value >> kPrecisionBits, 0, 255));
}

// Rows [offset, offset + out.height) of `in`, resampled along x.
void horizontal(const Rgb8Image& in, Rgb8Image& out, int offset, const Coefficients& c) {
    for (int yy = 0; yy < out.height; ++yy) {
        const uint8_t* line = in.pixels.data() + static_cast<size_t>(yy + offset) * in.width * 3;
        uint8_t* target = out.pixels.data() + static_cast<size_t>(yy) * out.width * 3;
        for (int xx = 0; xx < out.width; ++xx) {
            const int xmin = c.bounds[static_cast<size_t>(xx) * 2];
            const int xmax = c.bounds[static_cast<size_t>(xx) * 2 + 1];
            const int32_t* k = c.fixed.data() + static_cast<size_t>(xx) * c.ksize;
            for (int channel = 0; channel < 3; ++channel) {
                int sum = 1 << (kPrecisionBits - 1);
                for (int x = 0; x < xmax; ++x)
                    sum += line[(x + xmin) * 3 + channel] * k[x];
                target[xx * 3 + channel] = clip8(sum);
            }
        }
    }
}

void vertical(const Rgb8Image& in, Rgb8Image& out, const Coefficients& c) {
    for (int yy = 0; yy < out.height; ++yy) {
        const int ymin = c.bounds[static_cast<size_t>(yy) * 2];
        const int ymax = c.bounds[static_cast<size_t>(yy) * 2 + 1];
        const int32_t* k = c.fixed.data() + static_cast<size_t>(yy) * c.ksize;
        uint8_t* target = out.pixels.data() + static_cast<size_t>(yy) * out.width * 3;
        for (int xx = 0; xx < out.width; ++xx)
            for (int channel = 0; channel < 3; ++channel) {
                int sum = 1 << (kPrecisionBits - 1);
                for (int y = 0; y < ymax; ++y)
                    sum +=
                        in.pixels[(static_cast<size_t>(y + ymin) * in.width + xx) * 3 + channel] *
                        k[y];
                target[xx * 3 + channel] = clip8(sum);
            }
    }
}

Rgb8Image blank(int height, int width) {
    Rgb8Image image;
    image.height = height;
    image.width = width;
    image.pixels.resize(static_cast<size_t>(height) * width * 3);
    return image;
}

Rgb8Image crop(const Rgb8Image& image, int top, int left, int height, int width) {
    auto out = blank(height, width);
    for (int y = 0; y < height; ++y)
        std::memcpy(out.pixels.data() + static_cast<size_t>(y) * width * 3,
                    image.pixels.data() + (static_cast<size_t>(top + y) * image.width + left) * 3,
                    static_cast<size_t>(width) * 3);
    return out;
}

int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

void append_normalized(const Rgb8Image& tile, const ImageProcessing& p, std::vector<float>& out) {
    const size_t plane = static_cast<size_t>(tile.height) * tile.width;
    const size_t base = out.size();
    out.resize(base + plane * 3);
    for (size_t i = 0; i < plane; ++i)
        for (int channel = 0; channel < 3; ++channel) {
            // numpy: float64 rescale cast to float32, then float32 (x - mean) / std.
            const float rescaled = static_cast<float>(
                static_cast<double>(tile.pixels[i * 3 + channel]) * p.rescale_factor);
            out[base + channel * plane + i] = (rescaled - p.mean[channel]) / p.std[channel];
        }
}

} // namespace

std::pair<int, int> resize_to_longest_edge(int height, int width, int longest_edge) {
    if (height <= 0 || width <= 0 || longest_edge <= 0)
        throw std::invalid_argument("SmolVLM image and target sizes must be positive");
    // Scale the longer side to longest_edge; round the other side up to even.
    double aspect = static_cast<double>(width) / height;
    if (width >= height) {
        width = longest_edge;
        height = static_cast<int>(width / aspect);
        if (height % 2 != 0)
            ++height;
    } else {
        height = longest_edge;
        width = static_cast<int>(height * aspect);
        if (width % 2 != 0)
            ++width;
    }
    height = std::max(height, 1);
    width = std::max(width, 1);
    // Then keep both sides at or below the processor's absolute bound.
    aspect = static_cast<double>(width) / height;
    if (width >= height && width > kMaxImageSize) {
        width = kMaxImageSize;
        height = static_cast<int>(width / aspect);
    } else if (height > width && height > kMaxImageSize) {
        height = kMaxImageSize;
        width = static_cast<int>(height * aspect);
    }
    return {std::max(height, 1), std::max(width, 1)};
}

std::pair<int, int> resize_for_tiles(int height, int width, int tile) {
    if (height <= 0 || width <= 0 || tile <= 0)
        throw std::invalid_argument("SmolVLM tile geometry must be positive");
    const double aspect = static_cast<double>(width) / height;
    if (width >= height) {
        width = ceil_div(width, tile) * tile;
        height = ceil_div(static_cast<int>(width / aspect), tile) * tile;
    } else {
        height = ceil_div(height, tile) * tile;
        width = ceil_div(static_cast<int>(height * aspect), tile) * tile;
    }
    return {height, width};
}

Rgb8Image resize_lanczos(const Rgb8Image& input, int height, int width) {
    if (height <= 0 || width <= 0 || input.height <= 0 || input.width <= 0 ||
        input.pixels.size() != static_cast<size_t>(input.height) * input.width * 3)
        throw std::invalid_argument(
            "SmolVLM resize needs a complete RGB image and a positive size");
    if (height == input.height && width == input.width)
        return input;
    const bool need_horizontal = width != input.width;
    const bool need_vertical = height != input.height;
    // Pillow runs the vertical pass first when the height shrinks by more than
    // twice what the width shrinks; each pass rounds to 8 bits.
    const bool horizontal_first =
        !((input.height - height) > 0 && (input.height - height) > (input.width - width) * 2);
    Coefficients vertical_c, horizontal_c;
    if ((need_horizontal && horizontal_first) || need_vertical)
        vertical_c = coefficients(input.height, height);
    int first_row = 0, last_row = input.height;
    if (need_horizontal) {
        if (horizontal_first) {
            first_row = vertical_c.bounds[0];
            last_row = vertical_c.bounds[static_cast<size_t>(height) * 2 - 2] +
                       vertical_c.bounds[static_cast<size_t>(height) * 2 - 1];
            for (int i = 0; i < height; ++i)
                vertical_c.bounds[static_cast<size_t>(i) * 2] -= first_row;
        }
        horizontal_c = coefficients(input.width, width);
    }
    Rgb8Image current = input;
    if (horizontal_first) {
        if (need_horizontal) {
            auto next = blank(last_row - first_row, width);
            horizontal(current, next, first_row, horizontal_c);
            current = std::move(next);
        }
        if (need_vertical) {
            auto next = blank(height, width);
            vertical(current, next, vertical_c);
            current = std::move(next);
        }
    } else {
        if (need_vertical) {
            auto next = blank(height, current.width);
            vertical(current, next, vertical_c);
            current = std::move(next);
        }
        if (need_horizontal) {
            auto next = blank(height, width);
            horizontal(current, next, 0, horizontal_c);
            current = std::move(next);
        }
    }
    return current;
}

TiledImage tile_image(const Rgb8Image& image, const ImageProcessing& p) {
    if (p.tile <= 0 || p.longest_edge <= 0 || p.image_seq_len <= 0)
        throw std::invalid_argument("Invalid SmolVLM image processing parameters");
    const auto [height, width] = resize_to_longest_edge(image.height, image.width, p.longest_edge);
    const auto resized = resize_lanczos(image, height, width);
    const auto [tiled_height, tiled_width] = resize_for_tiles(height, width, p.tile);
    const auto grid = resize_lanczos(resized, tiled_height, tiled_width);
    TiledImage result;
    if (tiled_height > p.tile || tiled_width > p.tile) {
        result.rows = ceil_div(tiled_height, p.tile);
        result.cols = ceil_div(tiled_width, p.tile);
        const int tile_height = ceil_div(tiled_height, result.rows);
        const int tile_width = ceil_div(tiled_width, result.cols);
        if (tile_height != p.tile || tile_width != p.tile)
            throw std::logic_error("SmolVLM tiles must be whole vision tiles");
        result.frames = result.rows * result.cols + 1;
        for (int row = 0; row < result.rows; ++row)
            for (int col = 0; col < result.cols; ++col)
                append_normalized(crop(grid, row * p.tile, col * p.tile, p.tile, p.tile), p,
                                  result.pixels);
        append_normalized(resize_lanczos(grid, p.tile, p.tile), p, result.pixels);
    } else {
        if (grid.height != p.tile || grid.width != p.tile)
            throw std::logic_error("SmolVLM unsplit image must be one whole vision tile");
        result.frames = 1;
        append_normalized(grid, p, result.pixels);
    }
    return result;
}

Rgb8Image to_rgb8(const internal::ImageView& view) {
    if (view.data == nullptr || view.height == 0 || view.width == 0 || view.channels != 3)
        throw std::invalid_argument("SmolVLM images must be nonempty 3-channel RGB");
    if (view.height > static_cast<uint32_t>(std::numeric_limits<int>::max() / 3) ||
        view.width > static_cast<uint32_t>(std::numeric_limits<int>::max() / 3))
        throw std::invalid_argument("SmolVLM image dimensions are too large");
    const size_t count = static_cast<size_t>(view.height) * view.width * 3;
    auto image = blank(static_cast<int>(view.height), static_cast<int>(view.width));
    if (view.format == internal::ImageFormat::UInt8) {
        if (view.byte_size != count)
            throw std::invalid_argument("SmolVLM UInt8 image byte size does not match its shape");
        std::memcpy(image.pixels.data(), view.data, count);
        return image;
    }
    if (view.format != internal::ImageFormat::Float32 || view.byte_size != count * sizeof(float))
        throw std::invalid_argument("SmolVLM Float32 image byte size does not match its shape");
    const auto* values = static_cast<const float*>(view.data);
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i]) || values[i] < 0.0F || values[i] > 1.0F)
            throw std::invalid_argument("SmolVLM Float32 pixels must lie in [0, 1]");
        // The processor works on 8-bit RGB; x / 255 inputs round back exactly.
        image.pixels[i] = static_cast<uint8_t>(std::lround(values[i] * 255.0F));
    }
    return image;
}

} // namespace trtmc::smolvlm
