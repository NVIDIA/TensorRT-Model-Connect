/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/molmo2/runtime/preprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace trtmc::molmo2 {
namespace {
std::pair<int, int> tiling(int height, int width) {
    std::vector<std::pair<int, int>> candidates;
    for (int h = 1; h <= 8; ++h)
        for (int w = 1; h * w <= 8; ++w)
            candidates.emplace_back(h, w);
    std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) {
        return std::make_pair(a.first * a.second, a.first) <
               std::make_pair(b.first * b.second, b.first);
    });
    std::pair<int, int> best{1, 1};
    float min_upscale = std::numeric_limits<float>::infinity();
    float max_downscale = -std::numeric_limits<float>::infinity();
    bool upscale = false;
    for (auto candidate : candidates) {
        const float hs = height == 112 ? std::numeric_limits<float>::infinity()
                                       : static_cast<float>(candidate.first * 266) /
                                             static_cast<float>(height - 112);
        const float ws = width == 112 ? std::numeric_limits<float>::infinity()
                                      : static_cast<float>(candidate.second * 266) /
                                            static_cast<float>(width - 112);
        const float scale = std::min(hs, ws);
        if (scale >= 1.0F && (!upscale || scale < min_upscale)) {
            best = candidate;
            min_upscale = scale;
            upscale = true;
        } else if (!upscale && scale > max_downscale) {
            best = candidate;
            max_downscale = scale;
        }
    }
    return best;
}

// torchvision's tensor bilinear resize: half-pixel coordinates, no antialias,
// rounding back to uint8 before rescaling and normalizing the RGB samples.
std::vector<float> resize(const float* source, int height, int width, int out_h, int out_w) {
    std::vector<float> result(static_cast<std::size_t>(out_h) * out_w * 3);
    const float sy = static_cast<float>(height) / out_h, sx = static_cast<float>(width) / out_w;
    for (int y = 0; y < out_h; ++y) {
        const float fy = std::max(0.0F, (y + 0.5F) * sy - 0.5F);
        const int y0 = std::min(static_cast<int>(fy), height - 1),
                  y1 = std::min(y0 + 1, height - 1);
        const float dy = fy - y0;
        for (int x = 0; x < out_w; ++x) {
            const float fx = std::max(0.0F, (x + 0.5F) * sx - 0.5F);
            const int x0 = std::min(static_cast<int>(fx), width - 1),
                      x1 = std::min(x0 + 1, width - 1);
            const float dx = fx - x0;
            for (int c = 0; c < 3; ++c) {
                const float top = source[(y0 * width + x0) * 3 + c] * (1 - dx) +
                                  source[(y0 * width + x1) * 3 + c] * dx;
                const float bottom = source[(y1 * width + x0) * 3 + c] * (1 - dx) +
                                     source[(y1 * width + x1) * 3 + c] * dx;
                const float value = std::nearbyint(top * (1 - dy) + bottom * dy);
                result[(static_cast<std::size_t>(y) * out_w + x) * 3 + c] =
                    (value / 255.0F - 0.5F) / 0.5F;
            }
        }
    }
    return result;
}

void append_patches(std::vector<float>& output, const std::vector<float>& image, int width, int y0,
                    int x0) {
    for (int py = 0; py < 27; ++py)
        for (int px = 0; px < 27; ++px)
            for (int dy = 0; dy < 14; ++dy)
                for (int dx = 0; dx < 14; ++dx)
                    for (int c = 0; c < 3; ++c)
                        output.push_back(
                            image[((y0 + py * 14 + dy) * width + x0 + px * 14 + dx) * 3 + c]);
}
} // namespace

ImageInputs preprocess(const float* source, int height, int width) {
    if (!source || height <= 0 || width <= 0 || height > 32768 || width > 32768)
        throw std::invalid_argument("Molmo2 requires a nonempty RGB image");
    std::vector<float> rgb(static_cast<std::size_t>(height) * width * 3);
    for (std::size_t i = 0; i < rgb.size(); ++i) {
        if (!std::isfinite(source[i]) || source[i] < 0 || source[i] > 1)
            throw std::invalid_argument("Molmo2 RGB samples must be finite values in [0,1]");
        rgb[i] = std::nearbyint(source[i] * 255.0F);
    }
    const auto [th, tw] = tiling(height, width);
    ImageInputs result;
    result.crops = 1 + th * tw;
    result.pixels.reserve(static_cast<std::size_t>(result.crops) * 729 * 588);
    append_patches(result.pixels, resize(rgb.data(), height, width, 378, 378), 378, 0, 0);
    const int resized_h = th * 266 + 112, resized_w = tw * 266 + 112;
    const auto high = resize(rgb.data(), height, width, resized_h, resized_w);
    for (int cy = 0; cy < th; ++cy)
        for (int cx = 0; cx < tw; ++cx)
            append_patches(result.pixels, high, resized_w, cy * 266, cx * 266);
    auto pool = [&](int gh, int gw, bool low) {
        for (int y = 0; y < (gh + 1) / 2; ++y)
            for (int x = 0; x < (gw + 1) / 2; ++x)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int py = 2 * y + dy, px = 2 * x + dx;
                        int index = -1;
                        if (py < gh && px < gw) {
                            if (low)
                                index = py * 27 + px;
                            else {
                                const int cy = std::clamp((py - 4) / 19, 0, th - 1),
                                          cx = std::clamp((px - 4) / 19, 0, tw - 1);
                                index =
                                    (1 + cy * tw + cx) * 729 + (py - cy * 19) * 27 + px - cx * 19;
                            }
                        }
                        result.pooling.push_back(index);
                    }
    };
    pool(27, 27, true);
    pool(th * 19 + 8, tw * 19 + 8, false);
    result.pooled_height = (th * 19 + 9) / 2;
    result.pooled_width = (tw * 19 + 9) / 2;
    return result;
}

std::string image_prompt(const ImageInputs& inputs) {
    std::string prompt = "<low_res_im_start>";
    for (int i = 0; i < 196; ++i)
        prompt += "<im_patch>";
    prompt += "<im_end><im_start>";
    for (int y = 0; y < inputs.pooled_height; ++y) {
        for (int x = 0; x < inputs.pooled_width; ++x)
            prompt += "<im_patch>";
        prompt += "<im_col>";
    }
    return prompt + "<im_end>";
}

bool is_image_token(std::int32_t token) {
    switch (token) {
    case 151936:
    case 151937:
    case 151938:
    case 151939:
    case 151940:
    case 151942:
    case 151943:
    case 151944:
        return true;
    default:
        return false;
    }
}

std::vector<float> attention_mask(const std::vector<std::int32_t>& tokens, int past) {
    if (past < 0)
        throw std::invalid_argument("Molmo2 past length cannot be negative");
    const int n = static_cast<int>(tokens.size()), total = past + 1 + n;
    std::vector<float> mask(static_cast<std::size_t>(n) * total, -1e30F);
    for (int q = 0; q < n; ++q)
        for (int k = 1; k < total; ++k) {
            const bool causal = k <= past + q + 1;
            const bool bidirectional =
                past == 0 && k > past && is_image_token(tokens[q]) && is_image_token(tokens[k - 1]);
            if (causal || bidirectional)
                mask[static_cast<std::size_t>(q) * total + k] = 0;
        }
    return mask;
}
} // namespace trtmc::molmo2
