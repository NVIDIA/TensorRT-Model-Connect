/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/media_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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

template <typename Callable>
void check_throws(Callable&& callable, const char* label) {
    try {
        callable();
        check(false, label);
    } catch (const std::invalid_argument&) {
    }
}

std::vector<float> pixels_from_u8(const std::vector<uint8_t>& values) {
    std::vector<float> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index)
        result[index] = static_cast<float>(values[index]) / 255.0F;
    return result;
}

bool pixels_equal_u8(const std::vector<float>& actual, const std::vector<uint8_t>& expected) {
    if (actual.size() != expected.size())
        return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != static_cast<float>(expected[index]) / 255.0F)
            return false;
    }
    return true;
}

bool values_near(const std::vector<float>& actual, const std::vector<float>& expected,
                 float tolerance) {
    if (actual.size() != expected.size())
        return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::abs(actual[index] - expected[index]) > tolerance)
            return false;
    }
    return true;
}

void test_first_keyframe_stretches_with_pillow_lanczos() {
    trtmc::MediaImageInput image;
    image.height = 2;
    image.width = 2;
    image.pixels = pixels_from_u8({255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255});

    const auto output = trtmc::minimax_h3_prepare_keyframe_image(image, 3, 3, true);
    const std::vector<uint8_t> expected = {
        255, 0,   0,   128, 128, 0, 0,   255, 0,   128, 0,   128, 128, 128,
        128, 128, 255, 128, 0,   0, 255, 128, 128, 255, 255, 255, 255,
    };
    check(output.height == 3 && output.width == 3, "H3 first keyframe adopts the target canvas");
    check(pixels_equal_u8(output.pixels, expected),
          "H3 first keyframe matches Pillow LANCZOS bytes");
}

trtmc::MediaImageInput patterned_image(int32_t height, int32_t width, int32_t y_scale) {
    trtmc::MediaImageInput image;
    image.height = height;
    image.width = width;
    std::vector<uint8_t> bytes;
    bytes.reserve(static_cast<std::size_t>(height) * width * 3U);
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            bytes.push_back(static_cast<uint8_t>(x * 40));
            bytes.push_back(static_cast<uint8_t>(y * y_scale));
            bytes.push_back(static_cast<uint8_t>((x + y) * 25));
        }
    }
    image.pixels = pixels_from_u8(bytes);
    return image;
}

void test_follower_keyframe_cover_resizes_then_center_crops() {
    const auto image = patterned_image(3, 5, 80);
    const auto output = trtmc::minimax_h3_prepare_keyframe_image(image, 2, 2, false);
    const std::vector<uint8_t> expected = {
        13, 22, 15, 80, 22, 57, 13, 138, 51, 80, 138, 93,
    };
    check(output.height == 2 && output.width == 2, "H3 follower keyframe keeps the target canvas");
    check(pixels_equal_u8(output.pixels, expected),
          "H3 follower keyframe matches Pillow cover-crop bytes");
}

void test_cover_geometry_uses_python_half_even_rounding() {
    auto image = patterned_image(4, 5, 60);
    for (std::size_t pixel = 0; pixel < image.pixels.size() / 3U; ++pixel) {
        const int32_t x = static_cast<int32_t>(pixel % 5U);
        const int32_t y = static_cast<int32_t>(pixel / 5U);
        image.pixels[pixel * 3U + 2U] = static_cast<float>((x + y) * 20) / 255.0F;
    }
    const auto output = trtmc::minimax_h3_prepare_keyframe_image(image, 2, 2, false);
    check(pixels_equal_u8(output.pixels, {32, 32, 27, 128, 32, 75, 32, 148, 65, 128, 148, 113}),
          "H3 cover resize matches Python round-half-even geometry");
}

void test_keyframe_identity_and_validation() {
    trtmc::MediaImageInput image{{0.1234F, 0.5F, 0.9876F}, 1, 1};
    const auto output = trtmc::minimax_h3_prepare_keyframe_image(image, 1, 1, true);
    check(output.pixels == image.pixels, "H3 canvas-sized keyframe is an exact float32 identity");

    auto invalid = image;
    invalid.pixels[1] = std::numeric_limits<float>::quiet_NaN();
    check_throws([&] { (void)trtmc::minimax_h3_prepare_keyframe_image(invalid, 1, 1, true); },
                 "H3 keyframe rejects non-finite pixels");
    check_throws([&] { (void)trtmc::minimax_h3_prepare_keyframe_image(image, 0, 1, true); },
                 "H3 keyframe rejects an invalid target canvas");
}

trtmc::MediaVideoInput one_pixel_video(const std::vector<float>& values, float fps) {
    trtmc::MediaVideoInput video;
    video.num_frames = static_cast<int32_t>(values.size());
    video.height = 1;
    video.width = 1;
    video.fps = fps;
    for (float value : values)
        video.pixels.insert(video.pixels.end(), {value, value + 0.01F, value + 0.02F});
    return video;
}

std::vector<float> red_frames(const trtmc::MediaVideoInput& video) {
    std::vector<float> result;
    for (int32_t frame = 0; frame < video.num_frames; ++frame)
        result.push_back(video.pixels[static_cast<std::size_t>(frame) * 3U]);
    return result;
}

void test_reference_video_frame_slots_match_diffusers() {
    const auto dropped = trtmc::minimax_h3_normalize_reference_video_fps(
        one_pixel_video({0.0F, 0.1F, 0.2F, 0.3F, 0.4F}, 30.0F));
    check(dropped.num_frames == 4 && dropped.fps == 24.0F,
          "H3 30 fps reference normalizes to four 24 fps slots");
    check(red_frames(dropped) == std::vector<float>({0.0F, 0.1F, 0.3F, 0.4F}),
          "H3 reference video drops the last frame landing on a duplicate slot");

    const auto repeated =
        trtmc::minimax_h3_normalize_reference_video_fps(one_pixel_video({0.25F, 0.75F}, 12.0F));
    check(repeated.num_frames == 4, "H3 12 fps reference normalizes to four 24 fps slots");
    check(red_frames(repeated) == std::vector<float>({0.25F, 0.25F, 0.75F, 0.75F}),
          "H3 reference video holds frames across skipped slots");
}

void test_reference_video_identity_and_validation() {
    auto video = one_pixel_video({0.25F, 0.75F}, 24.0F);
    trtmc::MultiChannelAudioResult soundtrack{{0.1F, -0.1F}, 2, 32000, 1};
    video.soundtrack = soundtrack;
    const auto output = trtmc::minimax_h3_normalize_reference_video_fps(video);
    const bool soundtrack_unchanged = output.soundtrack.has_value() &&
                                      output.soundtrack->samples == soundtrack.samples &&
                                      output.soundtrack->num_samples == soundtrack.num_samples &&
                                      output.soundtrack->sample_rate == soundtrack.sample_rate &&
                                      output.soundtrack->num_channels == soundtrack.num_channels;
    check(output.pixels == video.pixels && soundtrack_unchanged,
          "H3 24 fps reference is an exact media identity");

    video.fps = std::numeric_limits<float>::infinity();
    check_throws([&] { (void)trtmc::minimax_h3_normalize_reference_video_fps(video); },
                 "H3 reference video rejects non-finite fps");
    video.fps = 24.0F;
    video.pixels[0] = 1.01F;
    check_throws([&] { (void)trtmc::minimax_h3_normalize_reference_video_fps(video); },
                 "H3 reference video rejects pixels outside the RGB range");
}

void test_reference_owned_canvas_geometry() {
    const auto square = trtmc::minimax_h3_resolve_reference_image_canvas(100, 100);
    check(square.height == 2048 && square.width == 2048,
          "H3 image reference uses a 2048-pixel short edge");
    const auto wide_image = trtmc::minimax_h3_resolve_reference_image_canvas(100, 400);
    check(wide_image.height == 2048 && wide_image.width == 8192,
          "H3 image reference has no area cap at the 4:1 boundary");

    const auto default_video = trtmc::minimax_h3_resolve_reference_video_canvas(9, 16);
    check(default_video.height == 768 && default_video.width == 1344,
          "H3 16:9 reference video resolves to the released canvas");
    const auto widest_video = trtmc::minimax_h3_resolve_reference_video_canvas(1, 4);
    check(widest_video.height == 512 && widest_video.width == 2016,
          "H3 video canvas applies the area cap before multiple-of-32 rounding");
    check_throws([&] { (void)trtmc::minimax_h3_resolve_reference_video_canvas(1, 5); },
                 "H3 reference canvas rejects aspect ratios wider than 4:1");

    check(trtmc::minimax_h3_trim_reference_num_frames(124) == 124,
          "H3 VAE keeps an aligned 124-frame reference");
    check(trtmc::minimax_h3_trim_reference_num_frames(123) == 107,
          "H3 VAE truncates to the largest 17*n+5 frame prefix");
}

void test_reference_float_media_quantizes_with_numpy_half_even_ties() {
    const float half_even_down = 10.5F / 255.0F;
    const float half_even_up = 11.5F / 255.0F;
    const auto quantized = trtmc::minimax_h3_quantize_reference_pixels(
        {0.1234F, half_even_down, half_even_up, 0.9876F});
    check(pixels_equal_u8(quantized, {31, 10, 12, 252}),
          "H3 in-memory references use NumPy-compatible uint8 half-even quantization");
}

void test_reference_audio_truncates_and_upmixes_mono() {
    trtmc::MultiChannelAudioResult audio{{-0.5F, 0.0F, 0.5F, 1.0F}, 4, 32000, 1};
    const auto output = trtmc::minimax_h3_prepare_reference_audio(audio, 0.0001);
    check(output.num_channels == 2 && output.num_samples == 3 && output.sample_rate == 32000,
          "H3 mono reference becomes truncated 32 kHz stereo");
    check(output.samples == std::vector<float>({-0.5F, 0.0F, 0.5F, -0.5F, 0.0F, 0.5F}),
          "H3 mono upmix repeats the channel without changing samples");
}

void test_reference_audio_preserves_channel_major_stereo() {
    trtmc::MultiChannelAudioResult audio{
        {-0.75F, -0.25F, 0.25F, 0.75F, 0.5F, 0.0F, -0.5F, -1.0F}, 4, 32000, 2};
    const auto output = trtmc::minimax_h3_prepare_reference_audio(audio, 1.0);
    check(output.samples == audio.samples,
          "H3 32 kHz stereo reference is a channel-major sample identity");
}

void test_reference_audio_vae_alignment_matches_hf_right_padding() {
    const trtmc::MultiChannelAudioResult audio{
        {-0.5F, 0.0F, 0.5F, 0.25F, 0.0F, -0.25F}, 3, 32000, 2};
    const auto aligned = trtmc::minimax_h3_align_reference_audio_for_vae(audio);
    check(aligned.num_samples == 800 && aligned.samples.size() == 1600,
          "H3 AudioVAE input rounds up to one 800-sample hop");
    check(std::equal(audio.samples.begin(), audio.samples.begin() + 3, aligned.samples.begin()) &&
              std::equal(audio.samples.begin() + 3, audio.samples.end(),
                         aligned.samples.begin() + 800),
          "H3 AudioVAE padding preserves both channel prefixes");
    check(std::all_of(aligned.samples.begin() + 3, aligned.samples.begin() + 800,
                      [](float value) { return value == 0.0F; }) &&
              std::all_of(aligned.samples.begin() + 803, aligned.samples.end(),
                          [](float value) { return value == 0.0F; }),
          "H3 AudioVAE alignment right-pads each channel with zeros");

    trtmc::MultiChannelAudioResult exact;
    exact.samples.assign(1600, 0.125F);
    exact.num_samples = 800;
    exact.sample_rate = 32000;
    exact.num_channels = 2;
    check(trtmc::minimax_h3_align_reference_audio_for_vae(exact).samples == exact.samples,
          "H3 AudioVAE-aligned audio remains an exact identity");
}

void test_reference_audio_matches_torchaudio_sinc_resampling() {
    trtmc::MultiChannelAudioResult source{{0.0F, 1.0F, 0.0F, -1.0F}, 4, 16000, 1};
    const auto output = trtmc::minimax_h3_prepare_reference_audio(source, 1.0);
    const std::vector<float> channel = {
        0.0042705992F, 0.54521817F,  0.99754024F,  0.80742592F,
        0.0F,          -0.80742592F, -0.99754024F, -0.54521817F,
    };
    std::vector<float> expected = channel;
    expected.insert(expected.end(), channel.begin(), channel.end());
    check(output.num_channels == 2 && output.num_samples == 8 && output.sample_rate == 32000,
          "H3 16 kHz reference resamples to the exact 32 kHz stereo geometry");
    check(values_near(output.samples, expected, 2.0e-6F),
          "H3 native sinc interpolation matches torchaudio default samples");

    const trtmc::MultiChannelAudioResult source_44100{
        {0.0F, 0.25F, -0.5F, 0.75F, -1.0F, 0.5F, 0.0F}, 7, 44100, 1};
    const auto downsampled = trtmc::minimax_h3_prepare_reference_audio(source_44100, 1.0);
    const std::vector<float> downsampled_channel = {
        0.12648456F, -0.09863649F, 0.14194083F, -0.35515428F, 0.25689617F, -0.05182421F,
    };
    expected = downsampled_channel;
    expected.insert(expected.end(), downsampled_channel.begin(), downsampled_channel.end());
    check(downsampled.num_samples == 6 && values_near(downsampled.samples, expected, 3.0e-6F),
          "H3 44.1 kHz downsampling matches torchaudio default samples");

    auto invalid = source;
    invalid.sample_rate = 32000;
    invalid.samples[0] = std::numeric_limits<float>::quiet_NaN();
    check_throws([&] { (void)trtmc::minimax_h3_prepare_reference_audio(invalid, 1.0); },
                 "H3 reference audio rejects non-finite samples");
    invalid = source;
    invalid.sample_rate = 32000;
    invalid.num_channels = 3;
    check_throws([&] { (void)trtmc::minimax_h3_prepare_reference_audio(invalid, 1.0); },
                 "H3 reference audio rejects unsupported channel counts");
}

void test_reference_audio_profile_covers_per_clip_hop_padding() {
    const std::vector<int32_t> sample_counts = {160004, 160004, 159992};
    int32_t latent_frames = 0;
    for (int32_t samples : sample_counts) {
        trtmc::MultiChannelAudioResult source;
        source.samples.assign(static_cast<std::size_t>(samples), 0.0F);
        source.num_samples = samples;
        source.sample_rate = 32000;
        source.num_channels = 1;
        const auto prepared = trtmc::minimax_h3_prepare_reference_audio(source, 124.0 / 24.0);
        const auto aligned = trtmc::minimax_h3_align_reference_audio_for_vae(prepared);
        latent_frames += aligned.num_samples / 800;
    }
    check(latent_frames == 602 && latent_frames * 2 == 1204,
          "H3 three-file 15-second audio budget includes independent hop padding");
}

} // namespace

int main() {
    test_first_keyframe_stretches_with_pillow_lanczos();
    test_follower_keyframe_cover_resizes_then_center_crops();
    test_cover_geometry_uses_python_half_even_rounding();
    test_keyframe_identity_and_validation();
    test_reference_video_frame_slots_match_diffusers();
    test_reference_video_identity_and_validation();
    test_reference_owned_canvas_geometry();
    test_reference_float_media_quantizes_with_numpy_half_even_ties();
    test_reference_audio_truncates_and_upmixes_mono();
    test_reference_audio_preserves_channel_major_stereo();
    test_reference_audio_vae_alignment_matches_hf_right_padding();
    test_reference_audio_matches_torchaudio_sinc_resampling();
    test_reference_audio_profile_covers_per_clip_hop_padding();
    return failures == 0 ? 0 : 1;
}
