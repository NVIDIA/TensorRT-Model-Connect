/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/conditioning.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

trtmc::VideoImageInput make_image(int32_t height, int32_t width, float value = 0.25F) {
    trtmc::VideoImageInput image;
    image.height = height;
    image.width = width;
    image.channels = 3;
    image.pixels.assign(static_cast<std::size_t>(height) * width * 3, value);
    return image;
}

void test_native_lanczos_contract() {
    const auto source = make_image(3, 5, 0.375F);
    const auto resized = trtmc::resize_minimax_h3_image_lanczos(source, 7, 2);
    check(resized.height == 7 && resized.width == 2 && resized.channels == 3,
          "H3 Lanczos returns packed target RGB geometry");
    bool constant_preserved = true;
    for (const float value : resized.pixels)
        constant_preserved = constant_preserved && std::abs(value - 0.375F) < 1.0e-5F;
    check(constant_preserved, "H3 Lanczos preserves a constant image");
}

void test_fl2va_anchor_and_crop_contract() {
    auto first = make_image(2, 4, 0.1F);
    auto last = make_image(2, 4, 0.0F);
    for (int32_t y = 0; y < 2; ++y) {
        for (int32_t x = 0; x < 4; ++x) {
            const float value = static_cast<float>(x) / 3.0F;
            for (int32_t channel = 0; channel < 3; ++channel)
                last.pixels[(static_cast<std::size_t>(y) * 4 + x) * 3 + channel] = value;
        }
    }

    const auto both = trtmc::prepare_minimax_h3_keyframes(first, last, 32, 32, 124);
    check(both.images.size() == 2 && both.anchors == std::vector<int32_t>({0, 123}),
          "H3 FL2VA preserves first/last packed anchors");
    check(both.images[0].height == 32 && both.images[0].width == 32,
          "H3 FL2VA stretches the geometry anchor to the canvas");
    check(both.images[1].height == 32 && both.images[1].width == 32,
          "H3 FL2VA cover-crops the follower to the canvas");
    check(both.images[1].pixels.front() > 0.1F && both.images[1].pixels.back() < 0.9F,
          "H3 FL2VA follower uses the centered portion of a wide source");

    const auto last_only = trtmc::prepare_minimax_h3_keyframes(
        std::nullopt, std::optional<trtmc::VideoImageInput>(last), 32, 32, 345);
    check(last_only.images.size() == 1 && last_only.anchors == std::vector<int32_t>({344}),
          "H3 FL2VA last-only input stays anchored to the final frame");

    const auto first_only = trtmc::prepare_minimax_h3_keyframes(
        std::optional<trtmc::VideoImageInput>(first), std::nullopt, 32, 32, 124);
    check(first_only.images.size() == 1 && first_only.anchors == std::vector<int32_t>({0}),
          "H3 FL2VA first-only input stays anchored to the first frame");
}

void test_ref2va_audio_contract() {
    trtmc::AudioResult mono;
    mono.sample_rate = 16000;
    mono.channels = 1;
    mono.samples.resize(16000);
    mono.num_samples = static_cast<int32_t>(mono.samples.size());
    for (std::size_t index = 0; index < mono.samples.size(); ++index)
        mono.samples[index] = std::sin(static_cast<float>(index) * 0.01F);

    const auto normalized = trtmc::normalize_minimax_h3_reference_audio(mono, 124);
    check(normalized.sample_rate == 32000 && normalized.channels == 2,
          "H3 Ref2VA audio is native 32 kHz stereo");
    check(normalized.samples.size() == 64000 && normalized.num_samples == 64000,
          "H3 Ref2VA audio is resampled exactly once");
    bool stereo_equal = true;
    for (std::size_t index = 0; index + 1 < normalized.samples.size(); index += 2)
        stereo_equal = stereo_equal && normalized.samples[index] == normalized.samples[index + 1];
    check(stereo_equal, "H3 Ref2VA mono references are duplicated into stereo");

    trtmc::AudioResult long_stereo;
    long_stereo.sample_rate = 32000;
    long_stereo.channels = 2;
    long_stereo.samples.assign(400000, 0.5F);
    long_stereo.num_samples = static_cast<int32_t>(long_stereo.samples.size());
    const auto truncated = trtmc::normalize_minimax_h3_reference_audio(long_stereo, 124);
    check(truncated.samples.size() == static_cast<std::size_t>((124LL * 32000 / 24) * 2),
          "H3 Ref2VA audio truncates at the source rate to generated duration");
}

void test_ref2va_frame_clock_contract() {
    const auto doubled = trtmc::make_minimax_h3_reference_frame_map(3, 12, 1, 124);
    check(doubled == std::vector<int32_t>({0, 0, 1, 1, 2, 2}),
          "H3 Ref2VA holds 12 fps frames onto the 24 fps clock");

    const auto ntsc = trtmc::make_minimax_h3_reference_frame_map(5, 30000, 1001, 124);
    check(ntsc == std::vector<int32_t>({0, 1, 3, 4}),
          "H3 Ref2VA preserves the reference whole-frame drop arithmetic");

    const auto truncated = trtmc::make_minimax_h3_reference_frame_map(10, 12, 1, 5);
    check(truncated == std::vector<int32_t>({0, 0, 1, 1, 2}),
          "H3 Ref2VA truncates after rate mapping to generated frame count");
}

void test_ref2va_order_and_limits() {
    trtmc::VideoReferenceInput audio;
    audio.kind = trtmc::VideoReferenceKind::kAudio;
    audio.audio.sample_rate = 32000;
    audio.audio.channels = 2;
    audio.audio.samples = {0.0F, 0.0F};
    audio.audio.num_samples = 2;
    const auto normalized_audio = trtmc::normalize_minimax_h3_references({audio}, 124);
    check(normalized_audio.size() == 1 &&
              normalized_audio.front().kind == trtmc::VideoReferenceKind::kAudio &&
              normalized_audio.front().audio.sample_rate == 32000 &&
              normalized_audio.front().audio.channels == 2,
          "H3 Ref2VA accepts and normalizes audio-only conditioning");

    std::vector<trtmc::VideoReferenceInput> too_many(13);
    bool rejected_total = false;
    try {
        (void)trtmc::normalize_minimax_h3_references(too_many, 124);
    } catch (const std::invalid_argument&) {
        rejected_total = true;
    }
    check(rejected_total, "H3 Ref2VA enforces the 12-entry public limit before decoding");
}

} // namespace

int main() {
    test_native_lanczos_contract();
    test_fl2va_anchor_and_crop_contract();
    test_ref2va_audio_contract();
    test_ref2va_frame_clock_contract();
    test_ref2va_order_and_limits();
    if (failures != 0)
        std::cerr << failures << " MiniMax-H3 conditioning test(s) failed\n";
    return failures == 0 ? 0 : 1;
}
