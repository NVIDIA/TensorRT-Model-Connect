/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/layoutlmv3/runtime/image_preprocess.h"

#include <ATen/ATen.h>
#include <cmath>
#include <stdexcept>

namespace trtmc::layoutlmv3 {
std::vector<float> image_pixels(const internal::ImageView& image) {
    if (!image.data || !image.height || !image.width || image.channels != 3 ||
        image.height > INT32_MAX || image.width > INT32_MAX)
        throw std::invalid_argument("LayoutLMv3 requires contiguous RGB pixels");
    const bool byte = image.format == internal::ImageFormat::UInt8;
    if ((!byte && image.format != internal::ImageFormat::Float32) ||
        image.byte_size !=
            static_cast<uint64_t>(image.height) * image.width * 3 * (byte ? 1 : sizeof(float)))
        throw std::invalid_argument("Invalid LayoutLMv3 RGB storage");
    if (!byte) {
        const auto* pixels = static_cast<const float*>(image.data);
        for (std::size_t i = 0; i < image.byte_size / sizeof(float); ++i)
            if (!std::isfinite(pixels[i]) || pixels[i] < 0 || pixels[i] > 1)
                throw std::invalid_argument("Invalid LayoutLMv3 RGB value");
    }

    const auto options = at::TensorOptions().device(at::kCPU).dtype(byte ? at::kByte : at::kFloat);
    auto pixels =
        at::from_blob(const_cast<void*>(image.data), {1, image.height, image.width, 3}, options)
            .to(at::kFloat);
    if (!byte)
        pixels = pixels * 255.0F;
    // Use the processor's separable antialiased bilinear operator. Its FP32
    // accumulation order matters to the subsequent document transformer.
    const auto channels = pixels.permute({0, 3, 1, 2}).contiguous();
    auto resized = at::_upsample_bilinear2d_aa(channels, {224, 224}, false);
    auto normalized = resized.sub(127.5F).div(127.5F).contiguous();
    const auto* values = normalized.const_data_ptr<float>();
    return {values, values + normalized.numel()};
}
} // namespace trtmc::layoutlmv3
