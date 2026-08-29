/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/pipeline.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
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

void check_near(float actual, float expected, float tolerance, const char* label) {
    check(std::abs(actual - expected) <= tolerance, label);
}

uint32_t float_bits(float value) {
    uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float float_from_bits(uint32_t bits) {
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void check_bits(float actual, uint32_t expected, const std::string& label) {
    if (float_bits(actual) != expected) {
        std::cerr << "FAIL: " << label << " expected_bits=0x" << std::hex << expected
                  << " actual_bits=0x" << float_bits(actual) << std::dec << '\n';
        ++failures;
    }
}

constexpr std::array<uint32_t, 50> kVideoSigmaBits = {
    0x3f800000U, 0x3f7f8e6bU, 0x3f7f186dU, 0x3f7e9dc1U, 0x3f7e1e1eU, 0x3f7d9937U, 0x3f7d0eb7U,
    0x3f7c7e3eU, 0x3f7be76dU, 0x3f7b49d0U, 0x3f7aa4f3U, 0x3f79f853U, 0x3f79435dU, 0x3f788576U,
    0x3f77bdeeU, 0x3f76ec07U, 0x3f760eeaU, 0x3f7525aaU, 0x3f742f43U, 0x3f732a8eU, 0x3f721643U,
    0x3f70f0f1U, 0x3f6fb8f9U, 0x3f6e6c83U, 0x3f6d097cU, 0x3f6b8d7eU, 0x3f69f5d3U, 0x3f683f56U,
    0x3f666666U, 0x3f6466c7U, 0x3f623b88U, 0x3f5fded5U, 0x3f5d49c4U, 0x3f5a740dU, 0x3f5753beU,
    0x3f53dcb0U, 0x3f4fffffU, 0x3f4bab23U, 0x3f46c6c6U, 0x3f413521U, 0x3f3acf91U, 0x3f336309U,
    0x3f2aaaaaU, 0x3f20473cU, 0x3f13b13aU, 0x3f042108U, 0x3ee0c7ceU, 0x3ead1207U, 0x3e4ccccdU,
    0x00000000U,
};

constexpr std::array<uint32_t, 49> kVideoTimestepBits = {
    0x00000000U, 0x3ae32a00U, 0x3b679300U, 0x3bb11f80U, 0x3bf0f100U, 0x3c19b240U, 0x3c3c5240U,
    0x3c607080U, 0x3c831260U, 0x3c96c600U, 0x3cab61a0U, 0x3cc0f5a0U, 0x3cd79460U, 0x3cef5140U,
    0x3d042120U, 0x3d113f90U, 0x3d1f1160U, 0x3d2da560U, 0x3d3d0bd0U, 0x3d4d5720U, 0x3d5e9bd0U,
    0x3d70f0f0U, 0x3d823838U, 0x3d8c9be8U, 0x3d97b420U, 0x3da39410U, 0x3db05168U, 0x3dbe0550U,
    0x3dccccd0U, 0x3ddcc9c8U, 0x3dee23c0U, 0x3e0084acU, 0x3e0ad8f0U, 0x3e162fccU, 0x3e22b108U,
    0x3e308d40U, 0x3e400004U, 0x3e515374U, 0x3e64e4e8U, 0x3e7b2b7cU, 0x3e8a60deU, 0x3e9939eeU,
    0x3eaaaaacU, 0x3ebf7188U, 0x3ed89d8cU, 0x3ef7bdf0U, 0x3f0f9c19U, 0x3f2976fcU, 0x3f4ccccdU,
};

constexpr std::array<uint32_t, 50> kAudioSigmaBits = {
    0x3f800000U, 0x3f7e3c07U, 0x3f7c6b6aU, 0x3f7a8d9eU, 0x3f78a211U, 0x3f76a82cU, 0x3f749f49U,
    0x3f7286bcU, 0x3f705dccU, 0x3f6e23b8U, 0x3f6bd7aeU, 0x3f6978d4U, 0x3f67063eU, 0x3f647ef0U,
    0x3f61e1e1U, 0x3f5f2df2U, 0x3f5c61f2U, 0x3f597c9bU, 0x3f567c8cU, 0x3f53604cU, 0x3f502649U,
    0x3f4cccceU, 0x3f495207U, 0x3f45b3f6U, 0x3f41f07cU, 0x3f3e0547U, 0x3f39efd4U, 0x3f35ad6bU,
    0x3f313b13U, 0x3f2c9592U, 0x3f27b960U, 0x3f22a2a2U, 0x3f1d4d1cU, 0x3f17b426U, 0x3f11d2a3U,
    0x3f0ba2e8U, 0x3f051eb8U, 0x3efc7e3fU, 0x3eedf8c9U, 0x3ede9bd2U, 0x3ece540eU, 0x3ebd0bd1U,
    0x3eaaaaaaU, 0x3e9714fcU, 0x3e822b63U, 0x3e579436U, 0x3e27904bU, 0x3de7d95cU, 0x3d70f0f1U,
    0x00000000U,
};

constexpr std::array<uint32_t, 49> kAudioTimestepBits = {
    0x00000000U, 0x3be1fc80U, 0x3c652580U, 0x3cae4c40U, 0x3cebbde0U, 0x3d157d40U, 0x3d360b70U,
    0x3d579440U, 0x3d7a2340U, 0x3d8ee240U, 0x3da14290U, 0x3db43960U, 0x3dc7ce10U, 0x3ddc0880U,
    0x3df0f0f8U, 0x3e034838U, 0x3e0e7838U, 0x3e1a0d94U, 0x3e260dd0U, 0x3e327ed0U, 0x3e3f66dcU,
    0x3e4cccc8U, 0x3e5ab7e4U, 0x3e693028U, 0x3e783e10U, 0x3e83f572U, 0x3e8c2058U, 0x3e94a52aU,
    0x3e9d89daU, 0x3ea6d4dcU, 0x3eb08d40U, 0x3ebababcU, 0x3ec565c8U, 0x3ed097b4U, 0x3edc5abaU,
    0x3ee8ba30U, 0x3ef5c290U, 0x3f01c0e0U, 0x3f09039cU, 0x3f10b217U, 0x3f18d5f9U, 0x3f217a18U,
    0x3f2aaaabU, 0x3f347582U, 0x3f3eea4eU, 0x3f4a1af2U, 0x3f561bedU, 0x3f6304d4U, 0x3f70f0f1U,
};

template <std::size_t SigmaCount, std::size_t TimestepCount>
void check_schedule_bits(const trtmc::MiniMaxH3Schedule& schedule,
                         const std::array<uint32_t, SigmaCount>& expected_sigmas,
                         const std::array<uint32_t, TimestepCount>& expected_timesteps,
                         const std::string& label) {
    check(schedule.sigmas.size() == expected_sigmas.size(), (label + " sigma count").c_str());
    check(schedule.timesteps.size() == expected_timesteps.size(),
          (label + " timestep count").c_str());
    if (schedule.sigmas.size() != expected_sigmas.size() ||
        schedule.timesteps.size() != expected_timesteps.size())
        return;
    for (std::size_t index = 0; index < expected_sigmas.size(); ++index)
        check_bits(schedule.sigmas[index], expected_sigmas[index],
                   label + " sigma " + std::to_string(index));
    for (std::size_t index = 0; index < expected_timesteps.size(); ++index)
        check_bits(schedule.timesteps[index], expected_timesteps[index],
                   label + " timestep " + std::to_string(index));
}

void test_pinned_schedules() {
    const auto video = trtmc::make_minimax_h3_schedule(50, 12.0F);
    const auto audio = trtmc::make_minimax_h3_schedule(50, 3.0F);
    check_schedule_bits(video, kVideoSigmaBits, kVideoTimestepBits,
                        "H3 shift-12 Diffusers schedule");
    check_schedule_bits(audio, kAudioSigmaBits, kAudioTimestepBits,
                        "H3 shift-3 Diffusers schedule");
}

void test_data_ward_euler_sign() {
    std::vector<float> sample = {1.0F, -2.0F};
    const std::vector<float> velocity = {0.5F, 0.25F};
    trtmc::minimax_h3_scheduler_step(sample.data(), velocity.data(), sample.size(), 0.25F, 0.75F,
                                     0.5F);
    check_near(sample[0], 1.125F, 1.0e-7F, "H3 Euler uses positive data-ward velocity");
    check_near(sample[1], -1.9375F, 1.0e-7F, "H3 Euler blend matches reference");
}

void test_euler_publishes_separate_fp32_operations() {
    std::vector<float> sample = {float_from_bits(0x3f477037U)};
    const std::vector<float> velocity = {float_from_bits(0x3ec784e5U)};
    trtmc::minimax_h3_scheduler_step(sample.data(), velocity.data(), sample.size(), 0.0F, 1.0F,
                                     float_from_bits(0x3f7e3c07U));
    check_bits(sample[0], 0x3f482057U,
               "H3 Euler matches separately published Diffusers FP32 operations");
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
    check(!request_rejected(audio_only), "H3 accepts an audio-only Ref2VA input");

    auto three_audio_clips = t2va;
    audio.audio = reference_audio(5, 1);
    three_audio_clips.references = {audio, audio, audio};
    check(!request_rejected(three_audio_clips),
          "H3 accepts three audio-only references totaling fifteen seconds");

    auto too_many_audio_clips = t2va;
    audio.audio = reference_audio(2, 1);
    too_many_audio_clips.references = {audio, audio, audio, audio};
    check(request_rejected(too_many_audio_clips),
          "H3 rejects more than three audio-only references");

    auto too_much_audio = t2va;
    audio.audio = reference_audio(6, 1);
    too_much_audio.references = {audio, audio, audio};
    check(request_rejected(too_much_audio),
          "H3 rejects audio-only references totaling more than fifteen seconds");

    auto too_short_audio = t2va;
    audio.audio = reference_audio(1, 1);
    too_short_audio.references = {audio};
    check(request_rejected(too_short_audio),
          "H3 rejects an audio-only reference shorter than two seconds");

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

void test_ref2va_audio_only_layout_has_no_visual_condition_rows() {
    const trtmc::MiniMaxH3PreparedReferenceLayout first{trtmc::AudioVideoReferenceKind::kAudio, 0,
                                                        0, 0, 2};
    const trtmc::MiniMaxH3PreparedReferenceLayout second{trtmc::AudioVideoReferenceKind::kAudio, 0,
                                                         0, 0, 3};
    const auto layout = trtmc::make_minimax_h3_ref2va_layout({1, 1}, {first, second}, 2, 2, 4, 3);

    check(layout.sequence_rows == 22,
          "H3 audio-only Ref2VA packs text, reference audio, and target media rows");
    check(layout.num_condition_video_rows == 0,
          "H3 audio-only Ref2VA does not claim synthetic visual condition rows");
    check(layout.num_condition_audio_rows == 10,
          "H3 audio-only Ref2VA counts both stereo reference blocks");
    check(layout.video_indices == std::vector<int32_t>({18, 19, 20, 21}),
          "H3 audio-only Ref2VA video indices contain only generated video rows");
    check(layout.audio_indices ==
              std::vector<int32_t>({2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}),
          "H3 audio-only Ref2VA preserves reference-audio order before target audio");

    const auto timestep_indices =
        trtmc::make_minimax_h3_conditioned_timestep_indices(layout, 0, 1, 2, 3);
    for (int32_t row = 2; row < 12; ++row)
        check(timestep_indices[static_cast<std::size_t>(row)] == 3,
              "H3 audio-only reference rows use the clean condition timestep");
    for (int32_t row = 12; row < 18; ++row)
        check(timestep_indices[static_cast<std::size_t>(row)] == 1,
              "H3 audio-only target audio rows use the live timestep");

    const auto position = [&](int32_t row, int32_t axis) {
        return layout.position_ids[static_cast<std::size_t>(row) * 3 + axis];
    };
    check_near(position(2, 0), 2.0F, 1.0e-6F, "H3 first audio-only reference starts after text");
    check_near(position(6, 0), 4.0F, 1.0e-6F,
               "H3 second audio-only reference follows the first block");
    check_near(position(12, 0), 7.0F, 1.0e-6F,
               "H3 target media follows both audio-only reference clocks");

    const auto reversed = trtmc::make_minimax_h3_ref2va_layout({1, 1}, {second, first}, 2, 2, 4, 3);
    check_near(reversed.position_ids[6 * 3], 3.0F, 1.0e-6F,
               "H3 audio-only reference order controls block boundaries");
    check(position(6, 0) != reversed.position_ids[6 * 3],
          "H3 audio-only references are not regrouped or reordered");
}

} // namespace

int main() {
    test_pinned_schedules();
    test_data_ward_euler_sign();
    test_euler_publishes_separate_fp32_operations();
    test_audio_rows_unpack_to_stereo_vae_batches();
    test_model_card_input_validation();
    test_fl2va_packed_layout_preserves_keyframe_and_media_clock();
    test_ref2va_packed_layout_preserves_ordered_reference_clock();
    test_ref2va_audio_only_layout_has_no_visual_condition_rows();
    return failures == 0 ? 0 : 1;
}
