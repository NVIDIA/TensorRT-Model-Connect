/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The vision engine reads patches in raster order and groups merge_size^2
// consecutive patches; Qwen2VLImageProcessor orders patches by merge group.

#include "families/qwen_image/runtime/qwen_image_vision_layout.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

// Each pixel stores channel * 1e6 + row * 1e3 + column of the source image.
std::vector<float> coordinate_image(int channels, int height, int width) {
    std::vector<float> image(static_cast<std::size_t>(channels) * height * width);
    for (int c = 0; c < channels; ++c)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                image[(static_cast<std::size_t>(c) * height + y) * width + x] =
                    static_cast<float>(c * 1000000 + y * 1000 + x);
    return image;
}

void test_raster_patches_follow_the_processor_order() {
    constexpr int channels = 6, patch = 2, merge = 2, height = 8, width = 12;
    const auto out = trtmc::qwen_image::to_merge_group_patch_order(
        coordinate_image(channels, height, width), channels, height, width, patch, merge);
    const int grid_w = width / patch;
    // The processor's patch k: (block row, block col, row in block, col in block).
    int k = 0;
    bool ok = true;
    for (int bh = 0; bh < height / patch / merge; ++bh)
        for (int bw = 0; bw < grid_w / merge; ++bw)
            for (int dh = 0; dh < merge; ++dh)
                for (int dw = 0; dw < merge; ++dw, ++k) {
                    const int src_y = (bh * merge + dh) * patch, src_x = (bw * merge + dw) * patch;
                    const int dst_y = k / grid_w * patch, dst_x = k % grid_w * patch;
                    for (int c = 0; c < channels; ++c)
                        for (int y = 0; y < patch; ++y)
                            for (int x = 0; x < patch; ++x) {
                                const float got =
                                    out[(static_cast<std::size_t>(c) * height + dst_y + y) * width +
                                        dst_x + x];
                                ok = ok && got == static_cast<float>(
                                                      c * 1000000 + (src_y + y) * 1000 + src_x + x);
                            }
                }
    check(ok, "raster patch k holds the processor's merge-group patch k");
    // Raster patch 2 (columns 4-5 of row 0) is the first merge group's
    // lower-left patch, which starts at source row 2, column 0.
    check(out[4] == 2000.0F, "third raster patch is the lower-left patch of the first merge group");
}

void test_rejects_misaligned_geometry() {
    bool threw = false;
    try {
        trtmc::qwen_image::to_merge_group_patch_order(std::vector<float>(6 * 6 * 6), 6, 6, 6, 2, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "dimensions not divisible by patch * merge are rejected");
}

} // namespace

int main() {
    test_raster_patches_follow_the_processor_order();
    test_rejects_misaligned_geometry();
    if (g_failures != 0) {
        std::cerr << g_failures << " qwen_image vision layout test(s) failed\n";
        return 1;
    }
    return 0;
}
