/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vitmatte/runtime/image_preprocess.h"

#include <array>
#include <iostream>
#include <limits>

int main() {
    // Frozen float32 outputs of the pinned VitMatteImageProcessor. Values next
    // to the midpoint distinguish rescale-then-normalize from (v-127.5)/127.5.
    constexpr std::array<int, 10> bytes{0, 1, 7, 63, 127, 128, 129, 201, 254, 255};
    constexpr std::array<float, 10> expected{
        -0x1p+0F,     -0x1.fbfbfcp-1F, -0x1.e3e3e4p-1F, -0x1.030304p-1F, -0x1.0101p-8F,
        0x1.0102p-8F, 0x1.8182p-7F,    0x1.272728p-1F,  0x1.fbfbfcp-1F,  0x1p+0F};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        const float normalized_input = static_cast<float>(bytes[i] / 255.0);
        if (trtmc::vitmatte::normalize_rgb(static_cast<float>(bytes[i]), true) != expected[i] ||
            trtmc::vitmatte::normalize_rgb(normalized_input, false) != expected[i]) {
            std::cerr << "ViTMatte reference pixel mismatch at byte " << bytes[i] << '\n';
            return 1;
        }
    }
    for (const bool byte : {false, true}) {
        for (const float invalid :
             {-1.0F, byte ? 256.0F : 1.01F, std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::quiet_NaN()}) {
            bool rejected = false;
            try {
                trtmc::vitmatte::normalize_rgb(invalid, byte);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            if (!rejected)
                return 1;
        }
    }
}
