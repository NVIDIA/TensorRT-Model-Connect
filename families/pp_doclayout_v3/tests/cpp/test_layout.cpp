/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/pp_doclayout_v3/runtime/layout.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace trtmc::pp_doclayout_v3;

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

int main() {
    const float rgb[] = {0.0F, 128.0F / 255, 1.0F};
    auto image = preprocess(rgb, 1, 1, 2);
    check(image.size() == 12 && image[0] == 0 && image[4] == rgb[1] && image[11] == 1,
          "bicubic resize must preserve constant RGB and produce CHW");
    const float stripe[] = {0, 1, 1, 0, 0, 1, 1, 0};
    std::vector<float> striped(8 * 8 * 3);
    for (int row = 0; row < 8; ++row)
        for (int column = 0; column < 8; ++column)
            for (int channel = 0; channel < 3; ++channel)
                striped[(row * 8 + column) * 3 + channel] = stripe[column];
    image = preprocess(striped.data(), 8, 8, 3);
    check(std::abs(image[0] - 218.0F / 255) < 1e-6F && image[1] == 0 &&
              std::abs(image[2] - 218.0F / 255) < 1e-6F,
          "resize must match torchvision uint8 bicubic with antialias disabled");
    std::vector<float> pattern(13 * 17 * 3);
    for (std::size_t i = 0; i < pattern.size(); ++i)
        pattern[i] = static_cast<float>((i * 37) % 256) / 255;
    image = preprocess(pattern.data(), 13, 17, 7);
    const int expected[3][7] = {{156, 159, 169, 178, 214, 217, 161},
                                {193, 162, 226, 215, 123, 12, 198},
                                {230, 199, 88, 252, 160, 49, 18}};
    for (int channel = 0; channel < 3; ++channel)
        for (int column = 0; column < 7; ++column)
            check(std::abs(image[(channel * 7 + 3) * 7 + column] -
                           expected[channel][column] / 255.0F) < 1e-6F,
                  "resize must preserve the reference integer coefficient rounding");
    bool rejected = false;
    try {
        (void)preprocess(nullptr, 1, 1, 2);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "empty image must fail");
    const float logits[] = {8, -8, -8, 3};
    const float boxes[] = {0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F};
    const float order[] = {-10000, -10, -10000, -10000};
    std::vector<float> masks(2 * 4 * 4, 10);
    auto result = decode(logits, boxes, order, masks.data(), 2, 2, 4, 16, 32, 32, 0.5F);
    check(result.regions.size() == 2 && result.regions[0].box.class_id == 1 &&
              result.regions[0].box.score < result.regions[1].box.score,
          "layout must use pairwise reading order rather than score order");
    check(result.regions[0].polygon.size() == 4 && result.image_height == 32 &&
              result.image_width == 32 && result.regions[0].box.x_min == 8,
          "layout must return original-image coordinates and contour polygons");
    std::fill(masks.begin(), masks.end(), -10);
    result = decode(logits, boxes, order, masks.data(), 2, 2, 4, 16, 32, 32, 0.5F);
    check(result.regions[0].polygon[2][0] == 24 && result.regions[0].polygon[2][1] == 24,
          "empty mask must fall back to the detection rectangle");
    // An L-shaped mask has five convex vertices after removing the concave turn.
    // A rectangle fallback cannot satisfy these exact contour coordinates.
    std::vector<float> nonrectangular(8 * 8, -10);
    for (int row = 0; row < 8; ++row)
        for (int column = 0; column < 8; ++column)
            if (row < 4 || column < 4)
                nonrectangular[row * 8 + column] = 10;
    const float full_box[] = {0.5F, 0.5F, 1, 1};
    auto contour =
        decode(logits, full_box, order, nonrectangular.data(), 1, 1, 8, 32, 32, 32, 0.5F);
    const std::vector<std::array<float, 2>> expected_polygon{
        {0, 0}, {0, 31}, {15, 31}, {31, 15}, {31, 0}};
    auto actual_polygon = contour.regions.at(0).polygon;
    std::rotate(actual_polygon.begin(),
                std::min_element(actual_polygon.begin(), actual_polygon.end()),
                actual_polygon.end());
    check(contour.regions.size() == 1 && actual_polygon == expected_polygon,
          "nonrectangular masks must produce their contour instead of the detection rectangle");
    const float below[] = {-1, -1, -1, -1};
    check(decode(below, boxes, order, masks.data(), 2, 2, 4, 16, 32, 32, 0.5F).regions.empty(),
          "no detections is a valid layout result");
    masks[0] = std::numeric_limits<float>::quiet_NaN();
    rejected = false;
    try {
        (void)decode(logits, boxes, order, masks.data(), 2, 2, 4, 16, 32, 32, 0.5F);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "nonfinite masks must fail before contour processing");
    std::cerr << "ALL PASSED\n";
}
