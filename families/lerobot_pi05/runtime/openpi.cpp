/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The fixed-point triangle filter and ALOHA linkage geometry follow the Apache-licensed
// Edge-LLM 0.12 Pi05Policy preprocessing implementation. The native path does not link Edge.
#include "families/lerobot_pi05/runtime/openpi.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace trtmc::lerobot_pi05 {
namespace {
//! Pillow's triangle resampler, one axis. The support widens with the downscale factor,
//! and the 8-bit passes accumulate in fixed point; a float accumulation lands a level off.
constexpr int32_t kPrecisionBits = 32 - 8 - 2;

struct ResampleAxis {
    std::vector<int32_t> weights; //!< \p taps entries per output pixel, scaled by 2^kPrecisionBits
    std::vector<int32_t> starts;
    int32_t taps{0};
};

ResampleAxis makeResampleAxis(int32_t srcSize, int32_t dstSize) {
    double const scale = static_cast<double>(srcSize) / dstSize;
    double const support = std::max(scale, 1.0);
    ResampleAxis axis;
    axis.taps = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
    axis.weights.assign(static_cast<size_t>(dstSize) * axis.taps, 0);
    axis.starts.assign(static_cast<size_t>(dstSize), 0);
    for (int32_t i = 0; i < dstSize; ++i) {
        double const center = (static_cast<double>(i) + 0.5) * scale;
        int32_t const begin = std::max(0, static_cast<int32_t>(center - support + 0.5));
        int32_t const end = std::min(srcSize, static_cast<int32_t>(center + support + 0.5));
        axis.starts[static_cast<size_t>(i)] = begin;
        auto const tapAt = [&](int32_t j) {
            return std::max(0.0, 1.0 - std::abs((static_cast<double>(j) + 0.5 - center) / support));
        };
        double total = 0.0;
        for (int32_t j = begin; j < end; ++j) {
            total += tapAt(j);
        }
        for (int32_t j = begin; total > 0.0 && j < end; ++j) {
            axis.weights[static_cast<size_t>(i) * axis.taps + (j - begin)] =
                static_cast<int32_t>(std::lround(tapAt(j) / total * (1 << kPrecisionBits)));
        }
    }
    return axis;
}

//! Pillow rounds each pass back to 8 bits before the next one reads it.
unsigned char round8(int32_t accumulated) {
    return static_cast<unsigned char>(std::clamp(accumulated >> kPrecisionBits, 0, 255));
}
} // namespace

static void resize_with_pad(unsigned char const* pixels, int32_t srcH, int32_t srcW, int32_t height,
                            int32_t width, float* out) {
    // Aspect-preserving target box, centred in the padded frame.
    double const ratio =
        std::max(static_cast<double>(srcW) / width, static_cast<double>(srcH) / height);
    auto const boxH = static_cast<int32_t>(static_cast<double>(srcH) / ratio);
    auto const boxW = static_cast<int32_t>(static_cast<double>(srcW) / ratio);
    int32_t const padTop = (height - boxH) / 2;
    int32_t const padLeft = (width - boxW) / 2;

    size_t const plane = static_cast<size_t>(height) * width;
    std::fill_n(out, 3 * plane, 0.0F); // black before the native graph applies SigLIP normalization
    if (boxH <= 0 || boxW <= 0) {
        return;
    }

    ResampleAxis const horizontal = makeResampleAxis(srcW, boxW);
    std::vector<unsigned char> rows(static_cast<size_t>(srcH) * boxW * 3);
    for (int32_t y = 0; y < srcH; ++y) {
        for (int32_t x = 0; x < boxW; ++x) {
            for (int32_t c = 0; c < 3; ++c) {
                int32_t acc = 1 << (kPrecisionBits - 1);
                for (int32_t k = 0; k < horizontal.taps; ++k) {
                    int32_t const j = horizontal.starts[static_cast<size_t>(x)] + k;
                    if (j >= srcW) {
                        break;
                    }
                    acc += horizontal.weights[static_cast<size_t>(x) * horizontal.taps + k] *
                           pixels[(static_cast<size_t>(y) * srcW + j) * 3 + c];
                }
                rows[(static_cast<size_t>(y) * boxW + x) * 3 + c] = round8(acc);
            }
        }
    }

    ResampleAxis const vertical = makeResampleAxis(srcH, boxH);
    for (int32_t y = 0; y < boxH; ++y) {
        for (int32_t x = 0; x < boxW; ++x) {
            for (int32_t c = 0; c < 3; ++c) {
                int32_t acc = 1 << (kPrecisionBits - 1);
                for (int32_t k = 0; k < vertical.taps; ++k) {
                    int32_t const j = vertical.starts[static_cast<size_t>(y)] + k;
                    if (j >= srcH) {
                        break;
                    }
                    acc += vertical.weights[static_cast<size_t>(y) * vertical.taps + k] *
                           rows[(static_cast<size_t>(j) * boxW + x) * 3 + c];
                }
                out[static_cast<size_t>(c) * plane + static_cast<size_t>(padTop + y) * width +
                    (padLeft + x)] = static_cast<float>(round8(acc)) / 255.0F;
            }
        }
    }
}

const std::uint8_t* openpi_rgb(const internal::ImageView& image, std::vector<std::uint8_t>& bytes) {
    const auto w = image.width, h = image.height;
    if (!w || !h || image.channels != 3 || !image.data ||
        w > std::numeric_limits<int32_t>::max() / 3U ||
        h > std::numeric_limits<int32_t>::max() / 3U)
        throw std::invalid_argument("openpi camera must be a nonempty RGB image");
    const bool floating = image.format == internal::ImageFormat::Float32;
    const std::size_t count = static_cast<std::size_t>(w) * h * 3;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        image.byte_size != count * (floating ? sizeof(float) : sizeof(std::uint8_t)))
        throw std::invalid_argument("openpi camera byte count disagrees with its shape");
    const double ratio = static_cast<double>(std::max(w, h)) / 224;
    if (w / ratio < 1 || h / ratio < 1)
        throw std::invalid_argument("openpi camera would resize to an empty dimension");
    auto* rgb = static_cast<const std::uint8_t*>(image.data);
    if (floating) {
        const auto* src = static_cast<const float*>(image.data);
        bytes.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(src[i]))
                throw std::invalid_argument("openpi camera has a non-finite pixel");
            bytes[i] = static_cast<std::uint8_t>(std::clamp(src[i], 0.0F, 1.0F) * 255.0F);
        }
        rgb = bytes.data();
    }
    return rgb;
}

void openpi_image(const internal::ImageView& image, float* pixels) {
    std::vector<std::uint8_t> bytes;
    const auto* rgb = openpi_rgb(image, bytes);
    resize_with_pad(rgb, image.height, image.width, 224, 224, pixels);
}

namespace {
constexpr float kAlohaJointFlip[]{1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
                                  1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
constexpr size_t kAlohaGripperDims[]{6U, 13U};

float gripperToAngular(float value) {
    float const linear = value * (0.05800F - 0.01844F) + 0.01844F;
    constexpr float kArmLength = 0.036F;
    constexpr float kHornRadius = 0.022F;
    float const cosine = (kHornRadius * kHornRadius + linear * linear - kArmLength * kArmLength) /
                         (2.0F * kHornRadius * linear);
    float const radians = std::asin(std::clamp(cosine, -1.0F, 1.0F));
    return (radians - 0.5476F) / (1.6296F - 0.5476F);
}

float gripperFromAngular(float value) {
    return (value + 0.5476F + 0.6213F) / (1.4910F + 0.6213F);
}

} // namespace

std::vector<float> openpi_state(std::string_view policy, Span<const float> state) {
    std::vector<float> out(state.begin(), state.end());
    if (!std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("openpi state must be finite");
    if (policy == "pi05_aloha") {
        if (out.size() != std::size(kAlohaJointFlip))
            throw std::invalid_argument("openpi ALOHA requires 14 state coordinates");
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] *= kAlohaJointFlip[i];
        for (auto i : kAlohaGripperDims)
            out[i] = gripperToAngular(out[i]);
    }
    return out;
}

void openpi_actions(std::string_view policy, const std::vector<float>& adapted,
                    std::vector<float>& actions) {
    if (policy != "pi05_aloha")
        return;
    constexpr auto width = std::size(kAlohaJointFlip);
    if (adapted.size() != width || actions.size() % width)
        throw std::runtime_error("Invalid ALOHA action or state dimensions");
    for (std::size_t i = 0; i < actions.size(); ++i) {
        auto col = i % width;
        if (col == 6 || col == 13)
            actions[i] = gripperFromAngular(actions[i] * kAlohaJointFlip[col]);
        else
            actions[i] = (actions[i] + adapted[col]) * kAlohaJointFlip[col];
    }
}
} // namespace trtmc::lerobot_pi05
