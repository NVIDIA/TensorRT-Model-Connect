/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/cohere_asr/runtime/frontend.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
int main() {
    const auto noise = trtmc::cohere_asr::normal_noise(56960);
    const float expected[] = {-0.597747862F, -0.218264416F, 0.171138376F,  2.244992256F,
                              -0.572005272F, -1.788945913F, -0.501267076F, -1.162263513F};
    for (int i = 0; i < 8; ++i)
        if (std::abs(noise[i] - expected[i]) > 1e-6F) {
            std::cerr << "dither differs from the official seeded float32 reference\n";
            return 1;
        }
    if (noise != trtmc::cohere_asr::normal_noise(56960))
        return 1;
    std::vector<float> audio(512, 0.0F), window(400, 1.0F), filters(128 * 257, 1.0F);
    auto values = trtmc::cohere_asr::features(audio.data(), audio.size(), window, filters);
    if (values.size() != 128 * 3)
        return 1;
    for (float value : values)
        if (!std::isfinite(value))
            return 1;
    audio[7] = std::numeric_limits<float>::quiet_NaN();
    try {
        (void)trtmc::cohere_asr::features(audio.data(), audio.size(), window, filters);
        return 1;
    } catch (const std::invalid_argument&) {
    }
    const std::vector<float> short_audio(511, 0.0F);
    try {
        (void)trtmc::cohere_asr::features(short_audio.data(), short_audio.size(), window, filters);
        return 1;
    } catch (const std::invalid_argument&) {
    }
    const std::vector<float> long_audio(560001, 0.0F);
    try {
        (void)trtmc::cohere_asr::features(long_audio.data(), long_audio.size(), window, filters);
        return 1;
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
