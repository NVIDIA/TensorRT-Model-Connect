/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/deepseek_ocr/runtime/image_preprocessor.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void verify(int width, int height, int left, int top, int right, int bottom) {
    trtmc::runtime::adapters::io::DecodedImage image;
    image.width = width;
    image.height = height;
    image.channels = 3;
    image.pixels.assign(width * height * 3, 255);
    trtmc::DeepseekOcrPreprocessConfig config;
    config.fixed_image_size = 16;
    config.preprocessor_type = "pad_center_chw";
    for (int channel = 0; channel < 3; ++channel) {
        config.image_mean[channel] = 0.5F;
        config.image_std[channel] = 0.5F;
    }
    const auto result = trtmc::deepseek_ocr_preprocess_decoded_image(image, config);
    if (!result.ok || result.pixel_values.size() != 3 * 16 * 16)
        throw std::runtime_error("padding must produce the declared CHW image");
    for (int channel = 0; channel < 3; ++channel) {
        for (int y = 0; y < 16; ++y) {
            for (int x = 0; x < 16; ++x) {
                const bool content = x >= left && x < right && y >= top && y < bottom;
                const float expected = content ? 1.0F : -1.0F / 255;
                const auto value = result.pixel_values[channel * 256 + y * 16 + x];
                if (std::abs(value - expected) > 1e-6F)
                    throw std::runtime_error("geometry or mean-color padding differs from PIL");
            }
        }
    }
}
} // namespace

int main() {
    try {
        // Independent PIL ImageOps.pad white-image bounding boxes at size16.
        verify(3, 7, 4, 0, 11, 16);
        verify(7, 3, 0, 4, 16, 11);
        verify(13, 16, 2, 0, 15, 16);
        verify(16, 13, 0, 2, 16, 15);
        verify(16, 16, 0, 0, 16, 16);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
