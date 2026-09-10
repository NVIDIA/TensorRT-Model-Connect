/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/hot_engine_policy.h"
#include "families/minimax_h3/runtime/pipeline.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
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

void test_shared_conditioning_activation_policy() {
    using namespace trtmc::minimax_h3;
    constexpr std::int64_t bundle_budget = 32LL << 30;
    constexpr std::int64_t tail_budget = 24LL << 30;
    for (const char* name : {"text_encoder_plan", "vision_encoder_plan"}) {
        check(uses_serial_execution_context(name),
              "H3 shared conditioning uses live-shape activation memory");
        for (bool retain : {false, true}) {
            check(!should_retain_hot_engine(name, retain),
                  "H3 conditioning activation policy does not retain extra engines");
            check(staged_plan_weight_streaming_budget(name, bundle_budget, retain, tail_budget) ==
                      bundle_budget,
                  "H3 conditioning activation policy preserves the weight-streaming budget");
        }
    }
    for (const char* name : {"denoiser_head_plan", "denoiser_tail_plan", "denoiser_finish_plan",
                            "ref2va_denoiser_plan", "ref2va_dit_head_plan", "ref2va_dit_tail_plan",
                            "ref2va_dit_finish_plan"}) {
        check(uses_serial_execution_context(name),
              "H3 existing denoiser live-shape activation policy is unchanged");
    }
    for (const char* name : {"adaln_precompute_plan", "ref2va_adaln_precompute_plan",
                            "fl2va_keyframe_vae_encoder_plan", "vae_tile_decoder_plan",
                            "audio_vae_decoder_plan", "video_super_resolution_plan",
                            "ref2va_shared_text_encoder_plan", "ref2va_shared_vision_encoder_plan",
                            "unknown_plan"}) {
        check(!uses_serial_execution_context(name),
              "H3 activation policy selects exact plan names, not timing labels or other stages");
    }
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

void test_first_block_cache_tail_schedule() {
    const auto schedule = trtmc::make_minimax_h3_schedule(50, 12.0F);
    const std::size_t forwards = schedule.timesteps.size();
    constexpr float threshold = 0.20F;
    check(schedule.sigmas.back() == 0.0F, "H3 dense schedule terminates at sigma zero");
    check(trtmc::should_compute_minimax_h3_tail(0, forwards, 0.0F, threshold),
          "H3 FirstBlockCache always computes the first tail");
    check(trtmc::should_compute_minimax_h3_tail(forwards - 1, forwards, 0.0F, threshold),
          "H3 FirstBlockCache refreshes the tail before the terminal sigma-to-zero update");
    check(!trtmc::should_compute_minimax_h3_tail(1, forwards, threshold, threshold),
          "H3 FirstBlockCache retains its strict threshold policy for interior steps");
    check(trtmc::should_compute_minimax_h3_tail(1, forwards, threshold + 0.01F, threshold),
          "H3 FirstBlockCache computes an interior tail above threshold");
    check(trtmc::should_compute_minimax_h3_tail(1, forwards,
                                                std::numeric_limits<float>::quiet_NaN(), threshold),
          "H3 FirstBlockCache computes an interior tail for a non-finite metric");
    check(trtmc::should_compute_minimax_h3_tail(1, forwards, 0.0F, 0.0F),
          "H3 FirstBlockCache zero threshold disables all interior reuse");
    check(trtmc::should_compute_minimax_h3_tail(1, forwards, std::numeric_limits<float>::infinity(),
                                                threshold),
          "H3 FirstBlockCache refreshes a non-finite residual baseline");
}

void test_data_ward_euler_sign() {
    std::vector<float> sample = {1.0F, -2.0F};
    const std::vector<float> velocity = {0.5F, 0.25F};
    trtmc::minimax_h3_scheduler_step(sample.data(), velocity.data(), sample.size(), 0.25F, 0.75F,
                                     0.5F);
    check_near(sample[0], 1.125F, 1.0e-7F, "H3 Euler uses positive data-ward velocity");
    check_near(sample[1], -1.9375F, 1.0e-7F, "H3 Euler blend matches reference");
}

void test_variable_text_position_layout() {
    constexpr int32_t media_rows = 414 + 37296;
    for (const int32_t text_rows : {1, 84, 218, 537, 2641}) {
        const auto positions = trtmc::make_minimax_h3_position_ids(text_rows);
        check(positions.size() == static_cast<std::size_t>(text_rows + media_rows) * 3,
              "H3 packed positions follow the actual text length");
        check_near(positions[static_cast<std::size_t>(text_rows) * 3],
                   static_cast<float>(text_rows), 0.0F,
                   "H3 audio rotary time starts after actual text rows");
        const auto video_start = static_cast<std::size_t>(text_rows + 414) * 3;
        check_near(positions[video_start], static_cast<float>(text_rows), 0.0F,
                   "H3 video rotary time starts after actual text rows");
    }

    for (const int32_t text_rows : {0, 2642}) {
        bool rejected = false;
        try {
            (void)trtmc::make_minimax_h3_position_ids(text_rows);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "H3 position layout rejects text rows outside its profile");
    }
}

void test_prompt_token_profile_boundaries() {
    for (const auto [tokens, maximum] : {std::pair<std::size_t, int32_t>{537, 537}, {2641, 2641}}) {
        try {
            trtmc::validate_minimax_h3_prompt_token_count(tokens, maximum);
        } catch (...) {
            check(false, "H3 prompt accepts the declared profile endpoint");
        }
    }

    for (const auto [tokens, maximum] : {std::pair<std::size_t, int32_t>{538, 537}, {2642, 2641}}) {
        bool rejected = false;
        try {
            trtmc::validate_minimax_h3_prompt_token_count(tokens, maximum);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "H3 prompt rejects one token beyond the declared profile");
    }
}

void test_denoiser_optimization_profile_selection() {
    using Layout = trtmc::MiniMaxH3DenoiserProfileLayout;
    const auto dynamic_layout = Layout::kFiveSecondDynamicThenPublicDynamic;
    const auto rejects = [](auto&& operation) {
        try {
            operation();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    const auto five_seconds = trtmc::make_minimax_h3_geometry(124, 768, 1344);
    const auto fifteen_seconds = trtmc::make_minimax_h3_geometry(345, 768, 1344);
    const auto max_canvas = trtmc::make_minimax_h3_geometry(124, 576, 1856);
    for (const auto& geometry : {
             five_seconds, fifteen_seconds,
             trtmc::make_minimax_h3_fl2va_geometry(five_seconds, 1),
             trtmc::make_minimax_h3_fl2va_geometry(five_seconds, 2),
             trtmc::make_minimax_h3_fl2va_geometry(fifteen_seconds, 2),
             trtmc::make_minimax_h3_fl2va_geometry(max_canvas, 2),
             trtmc::make_minimax_h3_geometry(124, 544, 960)}) {
        for (const int32_t text_rows : {1, 536, 537, 2641}) {
            for (const int32_t profile_count : {1, 2, 3}) {
                check(trtmc::select_minimax_h3_denoiser_profile(
                          profile_count, text_rows, geometry) == profile_count - 1,
                      "H3 always selects the final broad profile regardless of prompt or video geometry");
            }
            check(trtmc::select_minimax_h3_denoiser_profile(
                      2, text_rows, geometry, dynamic_layout) == 1,
                  "H3 historical two-dynamic-profile bundles always select their broad profile");
        }
    }
    check(trtmc::parse_minimax_h3_denoiser_profile_layout(
              "five_second_dynamic_then_public_dynamic", 2) == dynamic_layout,
          "H3 recognizes historical two-dynamic-profile metadata for compatibility");
    for (const int32_t profile_count : {1, 2, 3}) {
        check(trtmc::parse_minimax_h3_denoiser_profile_layout("", profile_count) == Layout::kLegacy,
              "H3 missing layout metadata retains legacy profile routing");
    }
    check(trtmc::parse_minimax_h3_denoiser_profile_layout("public_dynamic", 1) == Layout::kLegacy &&
              trtmc::parse_minimax_h3_denoiser_profile_layout(
                  "five_second_reference_then_public_dynamic", 2) == Layout::kLegacy &&
              trtmc::parse_minimax_h3_denoiser_profile_layout(
                  "five_second_t2va_then_fl2va_then_public_dynamic", 3) == Layout::kLegacy,
          "H3 recognizes released legacy profile-layout metadata");
    for (const int32_t profile_count : {0, 4}) {
        check(rejects([&] {
            (void)trtmc::select_minimax_h3_denoiser_profile(profile_count, 537, five_seconds);
        }), "H3 denoiser rejects an unsupported optimization-profile count");
    }
    for (const int32_t profile_count : {1, 3}) {
        check(rejects([&] {
            (void)trtmc::parse_minimax_h3_denoiser_profile_layout(
                "five_second_dynamic_then_public_dynamic", profile_count);
        }), "H3 rejects historical two-dynamic-profile metadata with a mismatched count");
    }
    check(rejects([] { (void)trtmc::parse_minimax_h3_denoiser_profile_layout("unknown", 2); }),
          "H3 rejects unknown profile-layout metadata");
    check(rejects([] {
        (void)trtmc::parse_minimax_h3_denoiser_profile_layout("public_dynamic", 2);
    }), "H3 rejects legacy profile-layout metadata with a mismatched count");
    for (const int32_t text_rows : {0, 2642}) {
        check(rejects([&] {
            (void)trtmc::select_minimax_h3_denoiser_profile(2, text_rows, five_seconds, dynamic_layout);
        }), "H3 dynamic profile rejects prompt lengths outside the public bounds");
    }
}

void test_public_video_geometry() {
    check(trtmc::align_minimax_h3_num_frames(120) == 124,
          "H3 aligns a requested five seconds to released causal-VAE geometry");
    check(trtmc::align_minimax_h3_num_frames(124) == 124,
          "H3 preserves an already aligned frame count");
    check(trtmc::align_minimax_h3_num_frames(344) == 345,
          "H3 aligns the longest supported request to 345 frames");

    const auto five_seconds = trtmc::make_minimax_h3_geometry(124, 768, 1344);
    check(five_seconds.video_latent_frames == 37, "H3 124f profile has 37 video latents");
    check(five_seconds.audio_latent_frames == 207, "H3 124f profile has 207 audio latents");
    check(five_seconds.audio_rows == 414, "H3 124f profile has two audio row streams");
    check(five_seconds.video_rows == 37296, "H3 124f profile has 37,296 video rows");

    const auto fifteen_seconds = trtmc::make_minimax_h3_geometry(345, 768, 1344);
    check(fifteen_seconds.video_latent_frames == 102,
          "H3 longest local profile has 102 video latents");
    check(fifteen_seconds.audio_latent_frames == 575,
          "H3 longest local profile has 575 audio latents");
    check(fifteen_seconds.audio_rows == 1150,
          "H3 longest local profile has two 575-row audio streams");
    check(fifteen_seconds.video_rows == 102816, "H3 longest local profile has 102,816 video rows");

    check(trtmc::make_minimax_h3_geometry(124, 768, 768).video_rows == 21312,
          "H3 accepts the public square 768p canvas");
    check(trtmc::make_minimax_h3_geometry(124, 1344, 768).video_rows == 37296,
          "H3 accepts the public portrait 9:16 canvas");

    for (const auto& canvas : std::array<std::array<int32_t, 2>, 2>{{{544, 960}, {960, 544}}}) {
        const auto short_geometry = trtmc::make_minimax_h3_geometry(124, canvas[0], canvas[1]);
        check(short_geometry.video_rows == 18870 && short_geometry.vae_tile_count == 15,
              "H3 124f profile accepts the documented 960x544 explicit canvas");
        const auto long_geometry = trtmc::make_minimax_h3_geometry(345, canvas[0], canvas[1]);
        check(long_geometry.video_rows == 52020 && long_geometry.vae_tile_count == 15,
              "H3 345f profile accepts the documented 960x544 explicit canvas");
    }

    for (const auto& canvas : std::array<std::array<int32_t, 2>, 1>{{{480, 864}}}) {
        check(trtmc::make_minimax_h3_geometry(124, canvas[0], canvas[1]).video_rows == 14985,
              "H3 compact five-second canvas fits the dynamic profile minimum");
        check(trtmc::make_minimax_h3_geometry(141, canvas[0], canvas[1]).video_rows == 17010,
              "H3 compact intermediate duration fits the dynamic profile");
        check(trtmc::make_minimax_h3_geometry(158, canvas[0], canvas[1]).video_rows == 19035,
              "H3 compact longer duration preserves its video row count");
        const auto geometry = trtmc::make_minimax_h3_geometry(345, canvas[0], canvas[1]);
        check(geometry.video_rows == 41310 && geometry.vae_tile_count == 15,
              "H3 345f profile accepts the compact super-resolution source canvas");
    }
    check(!trtmc::is_minimax_h3_native_canvas(864, 480),
          "H3 rejects portrait compact canvas without a portrait SR ABI");

    const auto max_rows = trtmc::make_minimax_h3_geometry(345, 576, 1856);
    check(max_rows.video_rows == 106488,
          "H3 dynamic row profile covers the largest rounded public canvas");

    for (const auto invalid_frames : {107, 360, 362}) {
        bool rejected = false;
        try {
            (void)trtmc::make_minimax_h3_geometry(invalid_frames, 768, 1344);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "H3 geometry rejects unsupported duration/frame shapes");
    }
}

void test_explicit_super_resolution_generation_contract() {
    trtmc::minimax_h3::SuperResolutionConfig sr;
    sr.enabled = true;
    sr.mode = "explicit";
    sr.section = "video_super_resolution_plan";
    sr.input_name = "frames";
    sr.output_name = "upscaled_frames";
    sr.source_height = 480;
    sr.source_width = 864;
    sr.target_height = 720;
    sr.target_width = 1296;
    sr.batch_min = 1;
    sr.batch_opt = 4;
    sr.batch_max = 8;
    const auto rejects = [](auto&& operation) {
        try {
            operation();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    trtmc::VideoGenerationRequest request;
    trtmc::VideoImageInput anchor;
    anchor.height = 1024;
    anchor.width = 1024;
    request.first_frame = anchor;
    for (const auto mode : {trtmc::VideoGenerationMode::kTextToVideoAudio,
                            trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio,
                            trtmc::VideoGenerationMode::kReferenceToVideoAudio}) {
        request.mode = mode;
        for (const int32_t frames : {124, 345}) {
            request.config.video_num_frames = frames;
            request.config.height = request.config.width = 0;
            const auto upscaled = trtmc::resolve_minimax_h3_generation(request, sr);
            check(upscaled.super_resolution && upscaled.geometry.output_height == 480 &&
                      upscaled.geometry.output_width == 864 &&
                      upscaled.geometry.output_frames == frames && upscaled.result_height == 720 &&
                      upscaled.result_width == 1296,
                  "H3 explicit SR resolves every workflow to fixed base and result geometry");
            request.config.height = 480;
            request.config.width = 864;
            const auto normal = trtmc::resolve_minimax_h3_generation(request);
            check(!normal.super_resolution && normal.result_height == 480 && normal.result_width == 864,
                  "H3 normal bundle never infers SR from compact request dimensions");
            const auto explicit_base = trtmc::resolve_minimax_h3_generation(request, sr);
            check(explicit_base.super_resolution && explicit_base.result_height == 720,
                  "H3 explicit SR accepts its declared source dimensions");
            request.config.height = 768;
            request.config.width = 1344;
            check(rejects([&] { (void)trtmc::resolve_minimax_h3_generation(request, sr); }),
                  "H3 explicit SR rejects a non-source canvas in every workflow");
        }
    }
    request.config = {};
    request.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
    check(trtmc::resolve_minimax_h3_generation(request).result_width == 768,
          "H3 normal FL2VA retains keyframe aspect resolution");
    request.mode = trtmc::VideoGenerationMode::kReferenceToVideoAudio;
    check(trtmc::resolve_minimax_h3_generation(request).result_width == 1344,
          "H3 normal REF2VA target geometry remains independent of references");
    request.config.height = 480;
    check(rejects([&] { (void)trtmc::resolve_minimax_h3_generation(request, sr); }),
          "H3 explicit SR rejects partial dimensions");
    for (int32_t field = 0; field < 7; ++field) {
        auto invalid = sr;
        if (field == 0) invalid.mode.clear();
        if (field == 1) invalid.mode = "automatic";
        if (field == 2) invalid.input_name.clear();
        if (field == 3) invalid.output_name = "wrong";
        if (field == 4) invalid.source_height = 768;
        if (field == 5) invalid.target_width = 1920;
        if (field == 6) invalid.batch_max = 16;
        check(rejects([&] { trtmc::minimax_h3::validate_super_resolution_config(invalid); }),
              "H3 explicit SR rejects legacy mode or mismatched native ABI metadata");
    }
}

void test_public_canvas_resolver_and_vae_tiles() {
    struct CanvasCase {
        double width;
        double height;
        int32_t expected_height;
        int32_t expected_width;
        int32_t tile_count;
    };
    constexpr std::array<CanvasCase, 6> public_ratios = {
        CanvasCase{21, 9, 672, 1536, 32}, CanvasCase{16, 9, 768, 1344, 28},
        CanvasCase{4, 3, 768, 1024, 20},  CanvasCase{1, 1, 768, 768, 16},
        CanvasCase{3, 4, 1024, 768, 20},  CanvasCase{9, 16, 1344, 768, 28},
    };
    for (const auto& expected : public_ratios) {
        const auto canvas = trtmc::resolve_minimax_h3_canvas(expected.width, expected.height);
        check(canvas.height == expected.expected_height && canvas.width == expected.expected_width,
              "H3 public aspect resolves to the Diffusers 768p canvas");
        for (const int32_t frames : {124, 345}) {
            const auto geometry =
                trtmc::make_minimax_h3_geometry(frames, canvas.height, canvas.width);
            check(geometry.vae_tile_count == expected.tile_count,
                  "H3 public canvas has the exact dynamic VAE tile batch");
            const auto positions = trtmc::make_minimax_h3_position_ids(84, geometry);
            check(positions.size() ==
                      static_cast<std::size_t>(84 + geometry.audio_rows + geometry.video_rows) * 3,
                  "H3 public canvas positions stay within the live packed rows");
        }
    }

    const auto landscape_limit = trtmc::resolve_minimax_h3_canvas(4, 1);
    const auto portrait_limit = trtmc::resolve_minimax_h3_canvas(1, 4);
    check(landscape_limit.height == 512 && landscape_limit.width == 2016,
          "H3 4:1 boundary keeps pre-round area semantics");
    check(portrait_limit.height == 2016 && portrait_limit.width == 512,
          "H3 1:4 boundary keeps pre-round area semantics");
    check(trtmc::make_minimax_h3_geometry(345, 512, 2016).vae_tile_count == 33,
          "H3 extreme public canvas reaches the 33-tile VAE profile maximum");

    const auto continuous_worst = trtmc::resolve_minimax_h3_canvas(3.631201, 1.0);
    check(continuous_worst.height == 544 && continuous_worst.width == 1952,
          "H3 continuous aspect resolver preserves its worst-case canvas");

    const auto default_tiles = trtmc::make_minimax_h3_vae_tile_layout(768, 1344);
    check(default_tiles.y_starts == std::vector<int32_t>({0, 160, 336, 512}) &&
              default_tiles.y_overlaps == std::vector<int32_t>({96, 80, 80}),
          "H3 dynamic VAE tiler exactly preserves supported 768p vertical tiles");
    check(default_tiles.x_starts == std::vector<int32_t>({0, 176, 352, 528, 704, 896, 1088}) &&
              default_tiles.x_overlaps == std::vector<int32_t>({80, 80, 80, 80, 64, 64}),
          "H3 dynamic VAE tiler exactly preserves supported 768p horizontal tiles");

    const auto extreme_tiles = trtmc::make_minimax_h3_vae_tile_layout(512, 2016);
    check(extreme_tiles.y_starts == std::vector<int32_t>({0, 128, 256}) &&
              extreme_tiles.x_starts.size() == 11 && extreme_tiles.x_starts.back() == 1760,
          "H3 VAE tiler covers the extreme canvas without gaps or overflow");

    for (const auto& invalid : std::array<std::array<int32_t, 2>, 5>{
             {{1024, 1024}, {768, 1408}, {480, 2048}, {800, 800}, {512, 2048}}}) {
        bool rejected = false;
        try {
            (void)trtmc::make_minimax_h3_geometry(124, invalid[0], invalid[1]);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "H3 geometry fails closed on a non-resolver canvas");
    }
    for (const auto invalid_ratio : {0.249999, 4.000001}) {
        bool rejected = false;
        try {
            (void)trtmc::resolve_minimax_h3_canvas(invalid_ratio, 1.0);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "H3 resolver rejects ratios outside the trained continuous boundary");
    }
}

void test_variable_duration_position_layout() {
    constexpr int32_t text_rows = 84;
    const auto geometry = trtmc::make_minimax_h3_geometry(345, 768, 1344);
    const auto positions = trtmc::make_minimax_h3_position_ids(text_rows, geometry);
    const auto sequence_rows = text_rows + geometry.audio_rows + geometry.video_rows;
    check(positions.size() == static_cast<std::size_t>(sequence_rows) * 3,
          "H3 15-second positions use the live packed row count");

    const std::size_t right_audio_start =
        static_cast<std::size_t>(text_rows + geometry.audio_latent_frames) * 3;
    check_near(positions[right_audio_start], static_cast<float>(text_rows), 0.0F,
               "H3 dynamic right-audio timestamps restart with the left stream");
    const std::size_t video_start = static_cast<std::size_t>(text_rows + geometry.audio_rows) * 3;
    check_near(positions[video_start], static_cast<float>(text_rows), 0.0F,
               "H3 dynamic video positions start after all live audio rows");
    const std::size_t last_video = static_cast<std::size_t>(sequence_rows - 1) * 3;
    check(positions[last_video] > positions[video_start],
          "H3 dynamic video time positions span every latent frame");
}

void test_fl2va_full_public_geometry_and_rotary_contract() {
    int32_t public_canvases = 0;
    for (int32_t height = 32; height <= 2016; height += 32) {
        for (int32_t width = 32; width <= 2016; width += 32) {
            bool official = true;
            trtmc::MiniMaxH3Geometry base;
            try {
                base = trtmc::make_minimax_h3_geometry(124, height, width);
            } catch (const std::invalid_argument&) {
                official = false;
            }
            if (!official)
                continue;
            ++public_canvases;
            for (int32_t frames = 124; frames <= 345; frames += 17) {
                base = trtmc::make_minimax_h3_geometry(frames, height, width);
                for (const int32_t keyframes : {1, 2}) {
                    const auto geometry = trtmc::make_minimax_h3_fl2va_geometry(base, keyframes);
                    const int32_t rows_per_frame =
                        (geometry.latent_height / 2) * (geometry.latent_width / 2);
                    check(geometry.condition_video_rows == keyframes * rows_per_frame &&
                              geometry.target_video_rows == base.video_rows &&
                              geometry.video_rows ==
                                  geometry.condition_video_rows + geometry.target_video_rows,
                          "H3 FL2VA couples condition and target row geometry");
                    check(geometry.video_rows <= 108576,
                          "H3 FL2VA stays inside the frozen video-row maximum");
                    const int32_t text_rows = keyframes == 1 ? 600 : 1200;
                    std::vector<int32_t> tags(static_cast<std::size_t>(text_rows), 1);
                    std::vector<int32_t> anchors = keyframes == 1
                                                       ? std::vector<int32_t>{frames - 1}
                                                       : std::vector<int32_t>{0, frames - 1};
                    const auto metadata =
                        trtmc::make_minimax_h3_fl2va_denoiser_metadata(tags, anchors, geometry);
                    const int32_t sequence_rows =
                        text_rows + geometry.audio_rows + geometry.video_rows;
                    check(sequence_rows <= 112367 &&
                              metadata.positions.size() ==
                                  static_cast<std::size_t>(sequence_rows) * 3,
                          "H3 FL2VA metadata exactly covers the frozen packed ABI");
                    const int32_t condition_begin = text_rows + geometry.audio_rows;
                    check(
                        metadata.timestep_indices[static_cast<std::size_t>(condition_begin)] == 2 &&
                            metadata.adaln_indices[static_cast<std::size_t>(condition_begin)] == 6,
                        "H3 FL2VA conditions select the near-clean AdaLN clock");
                    const int32_t target_begin = condition_begin + geometry.condition_video_rows;
                    check(metadata.timestep_indices[static_cast<std::size_t>(target_begin)] == 0,
                          "H3 FL2VA target video stays on the generated-video clock");
                    if (anchors.front() == frames - 1) {
                        check(metadata.positions[static_cast<std::size_t>(condition_begin) * 3] >
                                  metadata.positions[static_cast<std::size_t>(target_begin) * 3],
                              "H3 last-only keyframe uses the final rotary anchor");
                    }
                }
            }
        }
    }
    check(public_canvases == 98,
          "H3 FL2VA validates 95 resolver canvases, both 544x960 orientations, and 480x864");
}

void test_audio_latent_unpack_and_denormalize() {
    constexpr int32_t frames = 2;
    std::vector<float> rows(static_cast<std::size_t>(2 * frames * 32), 0.0F);
    rows[0] = 1.0F;
    rows[32] = 2.0F;
    rows[64] = 3.0F;
    rows[96] = 4.0F;
    const auto decoded = trtmc::unpack_and_denormalize_minimax_h3_audio(rows, frames);
    check(decoded.size() == rows.size(), "H3 audio unpack preserves scalar count");
    check_near(decoded[0], -0.0202116875F + 1.6895524263F, 1.0e-6F,
               "H3 audio denormalizes left frame zero");
    check_near(decoded[1], -0.0202116875F + 2.0F * 1.6895524263F, 1.0e-6F,
               "H3 audio transpose keeps left frame order");
    check_near(decoded[64], -0.0202116875F + 3.0F * 1.6895524263F, 1.0e-6F,
               "H3 audio unpack starts the right channel after left channel latents");
    check_near(decoded[65], -0.0202116875F + 4.0F * 1.6895524263F, 1.0e-6F,
               "H3 audio transpose keeps right frame order");
}

void test_audio_decoder_channel_duplication() {
    constexpr int32_t frames = 3;
    constexpr std::size_t channel_values = 32U * frames;
    std::vector<float> channel_major(2U * channel_values);
    for (std::size_t index = 0; index < channel_values; ++index) {
        channel_major[index] = static_cast<float>(index);
        channel_major[channel_values + index] = static_cast<float>(1000U + index);
    }

    for (int32_t channel = 0; channel < 2; ++channel) {
        const auto duplicated =
            trtmc::duplicate_minimax_h3_audio_decoder_channel(channel_major, frames, channel);
        check(duplicated.size() == channel_major.size(),
              "audio decoder duplication preserves the batch-two shape");
        for (std::size_t index = 0; index < channel_values; ++index) {
            const float expected =
                channel_major[static_cast<std::size_t>(channel) * channel_values + index];
            check(duplicated[index] == expected, "audio decoder duplication fills batch item zero");
            check(duplicated[channel_values + index] == expected,
                  "audio decoder duplication fills batch item one");
        }
    }

    bool bad_channel_threw = false;
    try {
        (void)trtmc::duplicate_minimax_h3_audio_decoder_channel(channel_major, frames, 2);
    } catch (const std::invalid_argument&) {
        bad_channel_threw = true;
    }
    check(bad_channel_threw, "audio decoder duplication rejects an invalid channel");
}

} // namespace

int main() {
    test_shared_conditioning_activation_policy();
    test_pinned_schedules();
    test_first_block_cache_tail_schedule();
    test_data_ward_euler_sign();
    test_variable_text_position_layout();
    test_prompt_token_profile_boundaries();
    test_denoiser_optimization_profile_selection();
    test_public_video_geometry();
    test_explicit_super_resolution_generation_contract();
    test_public_canvas_resolver_and_vae_tiles();
    test_variable_duration_position_layout();
    test_fl2va_full_public_geometry_and_rotary_contract();
    test_audio_latent_unpack_and_denormalize();
    test_audio_decoder_channel_duplication();
    return failures == 0 ? 0 : 1;
}
