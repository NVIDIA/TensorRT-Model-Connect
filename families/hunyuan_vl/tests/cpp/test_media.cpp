/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/hunyuan_vl/runtime/media.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace trtmc::hunyuan_vl;
int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        ++failures;
    }
}
ImageSettings settings(int64_t min_pixels = 262144) {
    ImageSettings s;
    s.patch = 16;
    s.merge = 2;
    s.min_pixels = min_pixels;
    s.max_pixels = 16777216;
    s.mean = {0.5F, 0.25F, 0.125F};
    s.std = {0.5F, 0.5F, 0.25F};
    s.rescale_factor = 1.0 / 255;
    return s;
}
ImagePlan plan(int height, int width, int max_patches = 65536, int64_t min_pixels = 262144) {
    static std::vector<uint8_t> pixels;
    pixels.assign(size_t(height) * width * 3, 0);
    trtmc::internal::ImageView view{pixels.data(),
                                    pixels.size(),
                                    uint32_t(height),
                                    uint32_t(width),
                                    3,
                                    trtmc::internal::ImageFormat::UInt8};
    return plan_image(view, settings(min_pixels), max_patches);
}
} // namespace

int main() {
    // Reference smart_resize values (factor 32, 512^2 to 4096^2 pixels).
    auto p = plan(1000, 700);
    expect(p.height == 992 && p.width == 704 && p.tokens() == 715, "plain rounding");
    p = plan(100, 50);
    expect(p.height == 736 && p.width == 384 && p.tokens() == 301, "minimum-area upscale");
    p = plan(528, 1040);
    expect(p.height == 512 && p.width == 1024 && p.tokens() == 530, "round half to even");
    p = plan(3508, 2480);
    expect(p.height == 3520 && p.width == 2496 && p.patches() == 34320, "A4 page");
    bool rejected = false;
    try {
        plan(5000, 5000, 16384);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "images above the built patch capacity are rejected");

    // Merge-block patch order, channel-major inside a patch, folded rescale + normalize.
    std::vector<uint8_t> pixels(64 * 64 * 3);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
            for (int c = 0; c < 3; ++c)
                pixels[(size_t(y) * 64 + x) * 3 + c] = uint8_t((y * 3 + x * 5 + c * 7) % 251);
    trtmc::internal::ImageView view{
        pixels.data(), pixels.size(), 64, 64, 3, trtmc::internal::ImageFormat::UInt8};
    const auto small = plan_image(view, settings(1), 16);
    expect(small.height == 64 && small.width == 64, "no resize at a factor multiple");
    const auto values = patchify(small, settings(1));
    const auto expected = [&](int y, int x, int c) {
        const float mean[] = {0.5F * 255, 0.25F * 255, 0.125F * 255};
        const float std[] = {0.5F * 255, 0.5F * 255, 0.25F * 255};
        return (float(pixels[(size_t(y) * 64 + x) * 3 + c]) - mean[c]) / std[c];
    };
    const size_t patch = 3 * 16 * 16;
    // Patch 1 is the right neighbour (dx = 1), patch 2 the one below (dy = 1), patch 4 the next
    // block.
    expect(std::abs(values[0] - expected(0, 0, 0)) < 1e-6F, "first value");
    expect(std::abs(values[patch + 256 + 3] - expected(0, 19, 1)) < 1e-6F, "dx neighbour");
    expect(std::abs(values[2 * patch + 2 * 256 + 17] - expected(17, 1, 2)) < 1e-6F, "dy neighbour");
    expect(std::abs(values[4 * patch] - expected(0, 32, 0)) < 1e-6F, "next merge block");

    // Equal grid and table sizes reproduce the table rows (class row skipped).
    std::vector<int32_t> ids;
    std::vector<float> lambdas;
    position_corners(small, 4, ids, lambdas);
    expect(ids.size() == 64 && lambdas.size() == 64, "four corners per patch");
    expect(ids[0] == 1 && lambdas[0] == 1 && lambdas[1] == 0 && lambdas[2] == 1 && lambdas[3] == 0,
           "identity interpolation");
    expect(ids[2 * 4] == 1 + 1 * 4 + 0 && ids[4 * 4] == 1 + 0 * 4 + 2,
           "patch order of position rows");
    return failures == 0 ? 0 : 1;
}
