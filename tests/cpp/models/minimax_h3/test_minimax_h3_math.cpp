/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/pipeline.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

void check_near(float actual, float expected, float tolerance, const char* label) {
    check(std::abs(actual - expected) <= tolerance, label);
}

void test_pinned_schedules() {
    const auto video = trtmc::make_minimax_h3_schedule(50, 12.0F);
    const auto audio = trtmc::make_minimax_h3_schedule(50, 3.0F);
    check(video.sigmas.size() == 50 && video.timesteps.size() == 49,
          "H3 video schedule uses 50 grid points and 49 evaluations");
    check(audio.sigmas.size() == 50 && audio.timesteps.size() == 49,
          "H3 audio schedule uses 50 grid points and 49 evaluations");
    check_near(video.sigmas[1], 0.998266875743866F, 1.0e-7F,
               "H3 shift-12 schedule matches Diffusers");
    check_near(audio.sigmas[1], 0.993103444576263F, 1.0e-7F,
               "H3 shift-3 schedule matches Diffusers");
    check_near(video.sigmas[48], 0.20000000298023224F, 1.0e-7F,
               "H3 video penultimate sigma matches Diffusers");
    check_near(audio.sigmas[48], 0.05882352963089943F, 1.0e-7F,
               "H3 audio penultimate sigma matches Diffusers");
}

void test_data_ward_euler_sign() {
    std::vector<float> sample = {1.0F, -2.0F};
    const std::vector<float> velocity = {0.5F, 0.25F};
    trtmc::minimax_h3_scheduler_step(sample.data(), velocity.data(), sample.size(), 0.25F, 0.75F,
                                     0.5F);
    check_near(sample[0], 1.125F, 1.0e-7F, "H3 Euler uses positive data-ward velocity");
    check_near(sample[1], -1.9375F, 1.0e-7F, "H3 Euler blend matches reference");
}

void test_audio_rows_unpack_to_stereo_vae_batches() {
    constexpr int channels = 2;
    constexpr int frames = 207;
    constexpr int latent_channels = 32;
    std::vector<float> rows(static_cast<std::size_t>(channels * frames * latent_channels));
    for (int channel = 0; channel < channels; ++channel) {
        for (int frame = 0; frame < frames; ++frame) {
            for (int latent_channel = 0; latent_channel < latent_channels; ++latent_channel) {
                const auto source =
                    static_cast<std::size_t>(channel * frames + frame) * latent_channels +
                    latent_channel;
                rows[source] = static_cast<float>(channel * 100000 + frame * 100 + latent_channel);
            }
        }
    }

    const auto latents = trtmc::minimax_h3_unpack_audio_latents(rows);
    check(latents.size() == rows.size(), "H3 audio unpack preserves every latent value");
    for (int channel = 0; channel < channels; ++channel) {
        for (int frame : {0, 103, 206}) {
            for (int latent_channel : {0, 17, 31}) {
                const auto target =
                    static_cast<std::size_t>(channel * latent_channels + latent_channel) * frames +
                    frame;
                const auto expected =
                    static_cast<float>(channel * 100000 + frame * 100 + latent_channel);
                check_near(latents[target], expected, 0.0F,
                           "H3 audio rows become channel-major VAE batches");
            }
        }
    }

    bool rejected = false;
    try {
        (void)trtmc::minimax_h3_unpack_audio_latents({0.0F});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "H3 audio unpack rejects the wrong fixed-profile row count");
}

trtmc::MediaImageInput one_pixel_image(float value = 0.5F) {
    return trtmc::MediaImageInput{{value, value, value}, 1, 1};
}

trtmc::MediaVideoInput reference_video(int frames, float fps) {
    trtmc::MediaVideoInput video;
    video.pixels.assign(static_cast<std::size_t>(frames) * 3, 0.5F);
    video.num_frames = frames;
    video.height = 1;
    video.width = 1;
    video.fps = fps;
    return video;
}

trtmc::MultiChannelAudioResult reference_audio(int samples, int sample_rate) {
    trtmc::MultiChannelAudioResult audio;
    audio.samples.assign(static_cast<std::size_t>(samples), 0.25F);
    audio.num_samples = samples;
    audio.sample_rate = sample_rate;
    audio.num_channels = 1;
    return audio;
}

bool request_rejected(const trtmc::AudioVideoRequest& request) {
    try {
        trtmc::validate_minimax_h3_request(request);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_model_card_input_validation() {
    trtmc::AudioVideoRequest t2va;
    t2va.prompt = "prompt";
    trtmc::validate_minimax_h3_request(t2va);

    auto fl2va = t2va;
    fl2va.first_image = one_pixel_image();
    fl2va.last_image = one_pixel_image();
    trtmc::validate_minimax_h3_request(fl2va);

    trtmc::AudioVideoReference image;
    image.kind = trtmc::AudioVideoReferenceKind::kImage;
    image.image = one_pixel_image();
    trtmc::AudioVideoReference video;
    video.kind = trtmc::AudioVideoReferenceKind::kVideo;
    video.video = reference_video(2, 1.0F);
    trtmc::AudioVideoReference audio;
    audio.kind = trtmc::AudioVideoReferenceKind::kAudio;
    audio.audio = reference_audio(2, 1);

    auto ref2va = t2va;
    ref2va.references = {image, video, audio};
    trtmc::validate_minimax_h3_request(ref2va);

    auto mixed_modes = ref2va;
    mixed_modes.first_image = one_pixel_image();
    check(request_rejected(mixed_modes), "H3 rejects keyframes mixed with omni-references");

    auto audio_only = t2va;
    audio_only.references = {audio};
    check(request_rejected(audio_only), "H3 rejects audio-only Ref2VA input");

    auto too_many_images = t2va;
    too_many_images.references.assign(10, image);
    check(request_rejected(too_many_images), "H3 enforces the nine-image reference cap");

    auto too_much_video = t2va;
    video.video = reference_video(8, 1.0F);
    too_much_video.references = {image, video, video};
    check(request_rejected(too_much_video),
          "H3 enforces the fifteen-second aggregate video-reference cap");

    auto too_much_soundtrack = t2va;
    video.video = reference_video(5, 1.0F);
    video.video.soundtrack = reference_audio(6, 1);
    too_much_soundtrack.references = {video, video, video};
    check(request_rejected(too_much_soundtrack),
          "H3 enforces the fifteen-second aggregate video-soundtrack cap");

    auto invalid_pixels = fl2va;
    invalid_pixels.first_image = one_pixel_image(-0.1F);
    check(request_rejected(invalid_pixels), "H3 rejects keyframe pixels outside [0,1]");

    auto invalid_guidance = t2va;
    invalid_guidance.config.guidance_scale = 1.0F;
    check(request_rejected(invalid_guidance),
          "H3 rejects guidance for guidance-distilled checkpoints");
}

void test_fl2va_packed_layout_preserves_keyframe_and_media_clock() {
    const auto layout = trtmc::make_minimax_h3_fl2va_layout(
        {1, 0, 1}, 2, 2, 4, 3,
        {trtmc::MiniMaxH3KeyframeAnchor::kFirst, trtmc::MiniMaxH3KeyframeAnchor::kLast});
    check(layout.sequence_rows == 17, "H3 FL2VA layout includes text, keyframes, audio, video");
    check(layout.num_condition_video_rows == 4 && layout.num_condition_audio_rows == 0,
          "H3 FL2VA layout counts only keyframe video conditioning rows");
    check(layout.text_indices == std::vector<int32_t>({0, 1, 2}),
          "H3 FL2VA text indices preserve the presentation order");
    check(layout.audio_indices == std::vector<int32_t>({7, 8, 9, 10, 11, 12}),
          "H3 FL2VA audio rows are channel-major");
    check(layout.video_indices == std::vector<int32_t>({3, 4, 5, 6, 13, 14, 15, 16}),
          "H3 FL2VA video indices put conditions before generated rows");
    check(layout.token_tags ==
              std::vector<int32_t>({1, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0}),
          "H3 FL2VA layout preserves text vision tags and media tags");

    const auto position = [&](int32_t row, int32_t axis) {
        return layout.position_ids[static_cast<std::size_t>(row) * 3 + axis];
    };
    check_near(position(3, 0), 3.0F, 1.0e-6F,
               "H3 first keyframe anchors at the media clock origin");
    check_near(position(5, 0), 3.0F + 20.0F / 3.0F, 1.0e-6F,
               "H3 last keyframe uses the reference temporal span");
    check_near(position(7, 0), 3.0F, 1.0e-6F, "H3 audio clock continues from text length");
    check_near(position(10, 0), 3.0F, 1.0e-6F,
               "H3 right audio channel restarts the shared time grid");
    check_near(position(13, 0), 3.0F, 1.0e-6F,
               "H3 generated video clock continues from text length");
    check_near(position(15, 0), 3.0F + 5.0F / 3.0F, 1.0e-6F,
               "H3 generated video follows the non-uniform temporal grid");

    bool rejected = false;
    try {
        (void)trtmc::make_minimax_h3_fl2va_layout({2}, 2, 2, 2, 3);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "H3 FL2VA layout rejects invalid conditioner tags");
}

void test_ref2va_packed_layout_preserves_ordered_reference_clock() {
    const trtmc::MiniMaxH3PreparedReferenceLayout image{trtmc::AudioVideoReferenceKind::kImage, 1,
                                                        2, 2, 0};
    const trtmc::MiniMaxH3PreparedReferenceLayout audio{trtmc::AudioVideoReferenceKind::kAudio, 0,
                                                        0, 0, 2};
    const trtmc::MiniMaxH3PreparedReferenceLayout video{trtmc::AudioVideoReferenceKind::kVideo, 2,
                                                        2, 4, 3};
    const auto layout =
        trtmc::make_minimax_h3_ref2va_layout({1, 1}, {image, audio, video}, 2, 2, 4, 3);
    check(layout.sequence_rows == 27, "H3 Ref2VA layout packs references and target rows");
    check(layout.num_condition_video_rows == 5 && layout.num_condition_audio_rows == 10,
          "H3 Ref2VA layout counts every reference media row");
    check(layout.text_indices == std::vector<int32_t>({0, 1}),
          "H3 Ref2VA text indices preserve the presentation");
    check(layout.audio_indices ==
              std::vector<int32_t>({3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 22}),
          "H3 Ref2VA audio indices preserve reference order before target audio");
    check(layout.video_indices == std::vector<int32_t>({2, 13, 14, 15, 16, 23, 24, 25, 26}),
          "H3 Ref2VA video indices preserve reference order before target video");

    const auto timestep_indices =
        trtmc::make_minimax_h3_conditioned_timestep_indices(layout, 0, 1, 2, 3);
    for (int32_t index = 0; index < layout.num_condition_video_rows; ++index)
        check(timestep_indices[static_cast<std::size_t>(layout.video_indices[index])] == 2,
              "H3 reference video rows use the frozen condition timestep");
    for (int32_t index = 0; index < layout.num_condition_audio_rows; ++index)
        check(timestep_indices[static_cast<std::size_t>(layout.audio_indices[index])] == 3,
              "H3 reference audio rows use the clean condition timestep");
    check(timestep_indices[static_cast<std::size_t>(layout.audio_indices.back())] == 1,
          "H3 generated audio rows use the live audio timestep");
    check(timestep_indices[static_cast<std::size_t>(layout.video_indices.back())] == 0,
          "H3 generated video rows use the live video timestep");

    const auto position = [&](int32_t row, int32_t axis) {
        return layout.position_ids[static_cast<std::size_t>(row) * 3 + axis];
    };
    check_near(position(2, 0), 2.0F, 1.0e-6F, "H3 image reference occupies one rotary clock unit");
    check_near(position(3, 0), 3.0F, 1.0e-6F,
               "H3 standalone audio follows the preceding image reference");
    check_near(position(7, 0), 5.0F, 1.0e-6F, "H3 video soundtrack shares its video origin");
    check_near(position(13, 0), 5.0F, 1.0e-6F, "H3 video reference shares its soundtrack origin");
    check_near(position(17, 0), 40.0F / 3.0F, 1.0e-6F,
               "H3 target media starts after the ordered reference clock");
    check_near(position(23, 0), 40.0F / 3.0F, 1.0e-6F,
               "H3 target audio and video remain synchronized");

    const auto reordered =
        trtmc::make_minimax_h3_ref2va_layout({1, 1}, {audio, image, video}, 2, 2, 4, 3);
    check_near(reordered.position_ids[2 * 3], 2.0F, 1.0e-6F,
               "H3 reordering references changes their rotary positions");
    check(position(3, 0) != reordered.position_ids[2 * 3],
          "H3 reference order is a semantic input, not regrouped by modality");
}

} // namespace

int main() {
    test_pinned_schedules();
    test_data_ward_euler_sign();
    test_audio_rows_unpack_to_stereo_vae_batches();
    test_model_card_input_validation();
    test_fl2va_packed_layout_preserves_keyframe_and_media_clock();
    test_ref2va_packed_layout_preserves_ordered_reference_clock();
    return failures == 0 ? 0 : 1;
}
