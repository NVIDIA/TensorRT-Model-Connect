/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Expected values come from OpenCV 4.13 cv2.resize(..., interpolation=cv2.INTER_AREA) on the same
// inputs; the camera path itself is checked bit for bit against LeRobot by preprocess_check.

#include "families/lerobot_groot/runtime/image.h"

#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    using namespace trtmc::lerobot_groot;

    // A 2x2 single-channel image enlarged to 3x3 and 5x5 by INTER_AREA.
    const std::vector<std::uint8_t> source = {0, 100, 200, 50};
    check(resize_area_up(source.data(), 2, 2, 1, 3, 3) ==
              std::vector<std::uint8_t>{0, 50, 100, 100, 88, 75, 200, 125, 50},
          "2x2 -> 3x3");
    check(resize_area_up(source.data(), 2, 2, 1, 5, 5) ==
              std::vector<std::uint8_t>{0,  0,  50,  100, 100, 0,  0,  50,  100, 100, 100, 100, 88,
                                        75, 75, 200, 200, 125, 50, 50, 200, 200, 125, 50,  50},
          "2x2 -> 5x5");
    // Same size is the identity, channels stay interleaved.
    std::vector<std::uint8_t> rgb(4 * 4 * 3);
    std::iota(rgb.begin(), rgb.end(), std::uint8_t{0});
    check(resize_area_up(rgb.data(), 4, 4, 3, 4, 4) == rgb, "identity");

    // A uniform camera normalises to one value in every patch row, with 32x32 merge blocks.
    std::vector<std::uint8_t> camera(64 * 64 * 3, 255);
    std::vector<float> patches;
    append_patches(camera.data(), 64, 60, patches);
    check(patches.size() == static_cast<std::size_t>(16) * kPatchValues, "patch rows");
    bool uniform = true;
    for (const float value : patches)
        uniform = uniform && value == 1.0F;
    check(uniform, "255 normalises to exactly 1");

    if (failures == 0)
        std::cout << "lerobot_groot image tests passed\n";
    return failures == 0 ? 0 : 1;
}
