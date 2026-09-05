/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sam2/runtime/sam2_pipeline.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

constexpr std::int32_t kHeight = 1280;
constexpr std::int32_t kWidth = 1088;
constexpr std::size_t kElements = static_cast<std::size_t>(kHeight) * kWidth * 3U;

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

bool rejects(const trtmc::VideoFrameView& frame) {
    try {
        (void)trtmc::sam2::convertSam2FloatFrameToRgb8(frame);
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

} // namespace

int main() {
    static_assert(std::is_abstract_v<trtmc::sam2::ISam2DeviceMaskSession>);
    static_assert(std::is_trivially_copyable_v<trtmc::sam2::Sam2DeviceMaskResultView>);
    trtmc::sam2::Sam2DeviceMaskResultView device_view;
    check(device_view.masks.size() == 5, "device result has exactly five mask pointers");
    check(device_view.mask_device_ordinal == -1, "empty device result has no device ordinal");

    std::vector<float> pixels(kElements, 0.0F);
    pixels[1] = 1.0F / 255.0F;
    pixels[2] = 0.5F;
    pixels[3] = 1.0F;
    trtmc::VideoFrameView frame{pixels.data(), pixels.size(), kHeight, kWidth,
                                trtmc::VideoFrameFormat::kRgbFloat32};

    const auto converted = trtmc::sam2::convertSam2FloatFrameToRgb8(frame);
    check(converted.size() == kElements, "conversion preserves the exact frame size");
    check(converted[0] == 0 && converted[1] == 1 && converted[2] == 128 && converted[3] == 255,
          "normalized float values round to RGB8");

    pixels[4] = -0.01F;
    check(rejects(frame), "negative input is rejected");
    pixels[4] = 1.01F;
    check(rejects(frame), "input above one is rejected");
    pixels[4] = std::numeric_limits<float>::quiet_NaN();
    check(rejects(frame), "non-finite input is rejected");
    pixels[4] = 0.0F;

    auto wrong_size = frame;
    --wrong_size.element_count;
    check(rejects(wrong_size), "wrong frame size is rejected");
    auto wrong_format = frame;
    wrong_format.format = trtmc::VideoFrameFormat::kRgb8;
    check(rejects(wrong_format), "non-float format is rejected by the converter");

    return failures == 0 ? 0 : 1;
}
