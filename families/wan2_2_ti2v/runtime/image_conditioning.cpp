/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wan2_2_ti2v/runtime/image_conditioning.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace trtmc::wan2_2_ti2v {

std::vector<float> first_frame_pixels(const internal::ImageView& image, int32_t height,
                                      int32_t width) {
    constexpr std::size_t channels = 3;
    const auto rows = static_cast<std::size_t>(height);
    const auto columns = static_cast<std::size_t>(width);
    const auto values = rows * columns * channels;
    const bool bytes = image.format == internal::ImageFormat::UInt8;
    if (height <= 0 || width <= 0 || image.data == nullptr || image.channels != channels ||
        image.height != rows || image.width != columns ||
        image.byte_size != values * (bytes ? sizeof(std::uint8_t) : sizeof(float))) {
        throw std::invalid_argument("Wan2.2 image-to-video needs a contiguous " +
                                    std::to_string(width) + "x" + std::to_string(height) +
                                    " RGB initial image, the bundle's video size; scale it to "
                                    "cover that size and center-crop it, as upstream i2v does");
    }
    std::vector<float> pixels(values);
    for (std::size_t index = 0; index < values; ++index) {
        const float value =
            bytes ? static_cast<float>(static_cast<const std::uint8_t*>(image.data)[index]) / 255.0F
                  : static_cast<const float*>(image.data)[index];
        if (!std::isfinite(value))
            throw std::invalid_argument("Wan2.2 initial image contains non-finite pixels");
        pixels[index % channels * rows * columns + index / channels] = (value - 0.5F) / 0.5F;
    }
    return pixels;
}

void keep_first_frame(std::vector<float>& latents, const std::vector<float>& first_frame,
                      int32_t latent_frames) {
    if (latent_frames <= 0 || first_frame.empty() ||
        latents.size() != first_frame.size() * static_cast<std::size_t>(latent_frames))
        throw std::invalid_argument("Wan2.2 first-frame latent shape is invalid");
    // CTHW: every channel holds latent_frames planes; frame 0 leads each channel.
    constexpr std::size_t channels = 48;
    if (first_frame.size() % channels != 0)
        throw std::invalid_argument("Wan2.2 first-frame latent must have 48 channels");
    const std::size_t plane = first_frame.size() / channels;
    for (std::size_t channel = 0; channel < channels; ++channel) {
        std::copy_n(first_frame.begin() + static_cast<std::ptrdiff_t>(channel * plane), plane,
                    latents.begin() + static_cast<std::ptrdiff_t>(channel * latent_frames * plane));
    }
}

} // namespace trtmc::wan2_2_ti2v
