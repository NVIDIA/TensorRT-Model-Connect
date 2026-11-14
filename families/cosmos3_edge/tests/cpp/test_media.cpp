/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/cosmos3_edge/runtime/media.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace
int main() {
    using namespace trtmc::cosmos3_edge;
    try {
        // 4 -> 2 antialiased bilinear uses three taps at the image boundary,
        // not the two taps of ordinary align_corners=false interpolation.
        Frame frame{{}, 32, 32, 16, 2};
        std::vector<float> weights;
        interpolate_positions(frame, 4, weights);
        const float axis[2][4]{{3.F / 7, 3.F / 7, 1.F / 7, 0}, {0, 1.F / 7, 3.F / 7, 3.F / 7}};
        require(weights.size() == 64, "interpolation matrix shape");
        for (int row = 0; row < 4; ++row)
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x)
                    require(std::abs(weights[row * 16 + y * 4 + x] -
                                     axis[row / 2][y] * axis[row % 2][x]) < 1e-7F,
                            "downsampling antialias weights");
        // A narrow valid frame needs much wider support than four 2D taps.
        frame.width = 2048;
        interpolate_positions(frame, 16, weights);
        for (int row = 0; row < frame.rows() * frame.cols(); ++row) {
            float sum = 0;
            int nonzero = 0;
            for (int i = 0; i < 256; ++i) {
                const float value = weights[row * 256 + i];
                require(std::isfinite(value) && value >= 0, "finite nonnegative position weights");
                sum += value;
                nonzero += value > 0;
            }
            require(std::abs(sum - 1) < 1e-6F, "position weights preserve constants");
            require(nonzero > 4, "downsampling support must not be truncated");
        }
        std::cout << "Cosmos3-Edge media contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
