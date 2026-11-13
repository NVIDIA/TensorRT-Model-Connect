/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wan2_2_ti2v/runtime/image_conditioning.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

template <typename Call>
void check_rejects(Call&& call, const std::string& message_part, const char* label) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        check(std::string(error.what()).find(message_part) != std::string::npos, label);
        return;
    }
    check(false, label);
}

trtmc::internal::ImageView bytes_view(const std::vector<std::uint8_t>& pixels, std::uint32_t height,
                                      std::uint32_t width) {
    return {pixels.data(), pixels.size(), height, width, 3, trtmc::internal::ImageFormat::UInt8};
}

void test_uint8_hwc_becomes_chw_in_minus_one_to_one() {
    // 2x3 RGB image, HWC: pixel (y, x) has channels (10 * y + x, 100 + x, 255 * y).
    std::vector<std::uint8_t> hwc;
    for (std::uint8_t y = 0; y < 2; ++y) {
        for (std::uint8_t x = 0; x < 3; ++x)
            hwc.insert(hwc.end(),
                       {static_cast<std::uint8_t>(10 * y + x), static_cast<std::uint8_t>(100 + x),
                        static_cast<std::uint8_t>(255 * y)});
    }
    const auto chw = trtmc::wan2_2_ti2v::first_frame_pixels(bytes_view(hwc, 2, 3), 2, 3);
    const auto expected = [](std::uint8_t value) { return (value / 255.0F - 0.5F) / 0.5F; };
    check(chw.size() == 18, "uint8 conversion keeps every value");
    check(chw[0 * 6 + 1 * 3 + 2] == expected(12), "channel 0 at (1, 2) is 10 * y + x");
    check(chw[1 * 6 + 0 * 3 + 1] == expected(101), "channel 1 at (0, 1) is 100 + x");
    check(chw[2 * 6 + 1 * 3 + 0] == 1.0F && chw[2 * 6 + 0] == -1.0F, "255 and 0 map to 1 and -1");
}

void test_float_pixels_match_their_uint8_values() {
    const std::vector<std::uint8_t> bytes = {0, 64, 128, 191, 255, 7};
    std::vector<float> floats;
    for (const auto value : bytes)
        floats.push_back(value / 255.0F);
    const trtmc::internal::ImageView view{floats.data(),
                                          floats.size() * sizeof(float),
                                          1,
                                          2,
                                          3,
                                          trtmc::internal::ImageFormat::Float32};
    check(trtmc::wan2_2_ti2v::first_frame_pixels(view, 1, 2) ==
              trtmc::wan2_2_ti2v::first_frame_pixels(bytes_view(bytes, 1, 2), 1, 2),
          "float32 [0, 1] pixels match their uint8 source");
}

void test_rejects_images_of_another_size_or_layout() {
    const std::vector<std::uint8_t> pixels(2 * 3 * 3, 0);
    check_rejects([&] { trtmc::wan2_2_ti2v::first_frame_pixels(bytes_view(pixels, 3, 2), 2, 3); },
                  "3x2 RGB initial image", "rejects transposed sizes, naming the bundle size");
    auto rgba = bytes_view(pixels, 2, 3);
    rgba.channels = 4;
    check_rejects([&] { trtmc::wan2_2_ti2v::first_frame_pixels(rgba, 2, 3); }, "RGB",
                  "rejects other channel counts");
    auto short_buffer = bytes_view(pixels, 2, 3);
    short_buffer.byte_size -= 1;
    check_rejects([&] { trtmc::wan2_2_ti2v::first_frame_pixels(short_buffer, 2, 3); }, "contiguous",
                  "rejects a short buffer");
    const std::vector<float> nan(3, std::numeric_limits<float>::quiet_NaN());
    const trtmc::internal::ImageView bad{nan.data(), nan.size() * sizeof(float),           1, 1,
                                         3,          trtmc::internal::ImageFormat::Float32};
    check_rejects([&] { trtmc::wan2_2_ti2v::first_frame_pixels(bad, 1, 1); }, "non-finite",
                  "rejects non-finite pixels");
}

void test_keeps_only_frame_zero_of_every_channel() {
    constexpr int frames = 3;
    constexpr int plane = 2;
    std::vector<float> latents(48 * frames * plane, 7.0F);
    std::vector<float> first(48 * plane);
    for (std::size_t index = 0; index < first.size(); ++index)
        first[index] = static_cast<float>(index);
    trtmc::wan2_2_ti2v::keep_first_frame(latents, first, frames);
    bool frame_zero = true;
    bool later_frames = true;
    for (int channel = 0; channel < 48; ++channel) {
        for (int frame = 0; frame < frames; ++frame) {
            for (int offset = 0; offset < plane; ++offset) {
                const float value = latents[(channel * frames + frame) * plane + offset];
                if (frame == 0)
                    frame_zero = frame_zero && value == first[channel * plane + offset];
                else
                    later_frames = later_frames && value == 7.0F;
            }
        }
    }
    check(frame_zero, "frame 0 of every channel is the encoded image");
    check(later_frames, "later frames are untouched");
    check_rejects([&] { trtmc::wan2_2_ti2v::keep_first_frame(latents, first, frames + 1); },
                  "shape", "rejects a frame count that does not match the latents");
}

} // namespace

int main() {
    test_uint8_hwc_becomes_chw_in_minus_one_to_one();
    test_float_pixels_match_their_uint8_values();
    test_rejects_images_of_another_size_or_layout();
    test_keeps_only_frame_zero_of_every_channel();
    return failures == 0 ? 0 : 1;
}
