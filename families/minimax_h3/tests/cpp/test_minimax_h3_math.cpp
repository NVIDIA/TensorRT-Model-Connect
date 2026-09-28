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
    for (const char* name : {"text_encoder_plan", "vision_encoder_plan", "text_encoder_1_plan",
                            "text_encoder_2_plan", "text_encoder_3_plan", "text_encoder_4_plan"}) {
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
    for (const char* name : {"denoiser_head_plan", "denoiser_tail_plan", "denoiser_tail_1_plan", "denoiser_finish_plan",
                            "ref2va_denoiser_plan", "ref2va_dit_head_plan", "ref2va_dit_tail_plan",
                            "ref2va_dit_finish_plan"}) {
        check(uses_serial_execution_context(name),
              "H3 existing denoiser live-shape activation policy is unchanged");
    }
    for (const char* name : {"adaln_precompute_plan", "adaln_precompute_1_plan", "ref2va_adaln_precompute_plan",
                            "fl2va_keyframe_vae_encoder_plan", "vae_tile_decoder_plan",
                            "audio_vae_decoder_plan", "video_super_resolution_plan",
                            "ref2va_shared_text_encoder_plan", "ref2va_shared_vision_encoder_plan",
                            "unknown_plan", "text_encoder_5_plan", "text_encoder_1_plan_extra"}) {
        check(!uses_serial_execution_context(name),
              "H3 activation policy selects exact plan names, not timing labels or other stages");
    }
    check(should_retain_hot_engine("denoiser_tail_1_plan", true) &&
              staged_plan_weight_streaming_budget("denoiser_tail_1_plan", bundle_budget, true,
                                                   tail_budget) == tail_budget,
          "H3 second tail retains the same serial streaming policy as the first");
    check(!should_retain_hot_engine("adaln_precompute_1_plan", true) &&
              staged_plan_weight_streaming_budget("adaln_precompute_1_plan", bundle_budget, true,
                                                   tail_budget) == bundle_budget,
          "H3 AdaLN segments are released between stages with the portable budget");
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

void test_first_block_cache_request_capacity() {
    struct Request {
        int32_t frames;
        int32_t height;
        int32_t width;
        int32_t text_rows;
        int32_t keyframes;
        std::size_t expected_bytes;
    };
    // One tensor; the runtime owns four with identical request-sized capacity.
    for (const auto& request : {
             Request{124, 768, 1344, 94, 0, 406468608U},
             Request{345, 768, 1344, 94, 0, 1118853120U},
             Request{124, 480, 864, 94, 0, 166580736U},
             Request{345, 480, 864, 94, 0, 457540608U},
             Request{124, 480, 864, 938, 2, 184364544U},
             Request{345, 480, 864, 938, 2, 475324416U},
             Request{124, 768, 1344, 2144, 2, 450186240U},
             Request{345, 768, 1344, 2144, 2, 1162570752U}}) {
        auto geometry = trtmc::make_minimax_h3_geometry(
            request.frames, request.height, request.width);
        if (request.keyframes != 0)
            geometry = trtmc::make_minimax_h3_fl2va_geometry(geometry, request.keyframes);
        check(trtmc::minimax_h3_cache_tensor_bytes(request.text_rows, geometry) ==
                  request.expected_bytes,
              "H3 cache capacity follows actual T2VA/FL2VA text, media, and condition rows");
    }

    const auto short_geometry = trtmc::make_minimax_h3_geometry(124, 480, 864);
    const auto minimum = trtmc::minimax_h3_cache_tensor_bytes(1, short_geometry);
    const auto longer_prompt = trtmc::minimax_h3_cache_tensor_bytes(2641, short_geometry);
    check(minimum == 15400U * 5376U * sizeof(uint16_t),
          "H3 cache accepts the broad dynamic profile minimum");
    check(longer_prompt - minimum == 2640U * 5376U * sizeof(uint16_t) &&
              trtmc::minimax_h3_cache_tensor_bytes(1, short_geometry) == minimum,
          "H3 cache byte requirement grows and shrinks with the current prompt");
    const auto maximum_geometry = trtmc::make_minimax_h3_fl2va_geometry(
        trtmc::make_minimax_h3_geometry(345, 576, 1856), 2);
    check(trtmc::minimax_h3_cache_tensor_bytes(2641, maximum_geometry) == 1208169984U,
          "H3 cache retains the complete public dynamic profile maximum");

    for (const int32_t text_rows : {0, 2642, std::numeric_limits<int32_t>::max()}) {
        bool rejected = false;
        try {
            (void)trtmc::minimax_h3_cache_tensor_bytes(text_rows, short_geometry);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "H3 cache rejects invalid prompt rows before allocation");
    }
    for (const auto [video_rows, audio_rows] : {
             std::pair<int32_t, int32_t>{14984, 414}, {108577, 414},
             {14985, 413}, {14985, 1151}, {std::numeric_limits<int32_t>::max(), 414}}) {
        auto invalid_geometry = short_geometry;
        invalid_geometry.video_rows = video_rows;
        invalid_geometry.audio_rows = audio_rows;
        bool rejected = false;
        try {
            (void)trtmc::minimax_h3_cache_tensor_bytes(94, invalid_geometry);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "H3 cache rejects per-modality profile overflow before allocation");
    }
}

void test_data_ward_euler_sign() {
    std::vector<float> sample = {1.0F, -2.0F};
    const std::vector<float> velocity = {0.5F, 0.25F};
    trtmc::minimax_h3_scheduler_step(sample.data(), velocity.data(), sample.size(), 0.25F, 0.75F,
                                     0.5F);
    check_near(sample[0], 1.125F, 1.0e-7F, "H3 Euler uses positive data-ward velocity");
    check_near(sample[1], -1.9375F, 1.0e-7F, "H3 Euler blend matches reference");
}

void test_turbo_sampler_contract() {
    using trtmc::MiniMaxH3Sampler;
    check(trtmc::parse_minimax_h3_sampler("") == MiniMaxH3Sampler::kDistilled &&
              trtmc::parse_minimax_h3_sampler("distilled") == MiniMaxH3Sampler::kDistilled,
          "H3 absent/explicit distilled sampler preserves the original schedule");
    check(trtmc::parse_minimax_h3_sampler("turbo_euler") == MiniMaxH3Sampler::kTurboEuler,
          "H3 Turbo sampler requires an explicit bundle contract");
    bool rejected = false;
    try { (void)trtmc::parse_minimax_h3_sampler("turbo"); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "H3 rejects unknown sampler aliases");
    trtmc::MiniMaxH3DenoiserConfig original;
    trtmc::validate_minimax_h3_sampler_config(original);
    auto turbo = original;
    turbo.sampler = MiniMaxH3Sampler::kTurboEuler;
    turbo.first_block_cache = false;
    turbo.scheduler_grid_points = 9;
    turbo.transformer_forwards = 8;
    trtmc::validate_minimax_h3_sampler_config(turbo);
    for (int invalid = 0; invalid < 5; ++invalid) {
        auto config = turbo;
        if (invalid == 0) config.first_block_cache = true;
        if (invalid == 1) config.scheduler_grid_points = 8;
        if (invalid == 2) config.transformer_forwards = 7;
        if (invalid == 3) config.guidance_scale = 2.0F;
        if (invalid == 4) config.sampler = MiniMaxH3Sampler::kDistilled;
        rejected = false;
        try { trtmc::validate_minimax_h3_sampler_config(config); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "H3 rejects inconsistent Turbo or original sampler contracts");
    }
}

void test_segmented_plan_contracts() {
    using namespace trtmc;
    MiniMaxH3DenoiserConfig config;
    check(config.text_encoder_sections == std::vector<std::string>{"text_encoder_plan"} &&
              config.adaln_precompute_sections == std::vector<std::string>{"adaln_precompute_plan"} &&
              config.denoiser_tail_sections == std::vector<std::string>{"denoiser_tail_plan"},
          "Original bundles retain their single-plan defaults");
    validate_minimax_h3_plan_sections({"text_encoder_plan", "text_encoder_1_plan"});
    for (const auto& invalid : std::vector<std::vector<std::string>>{
             {}, {""}, {"_plan"}, {"a_plan", "a_plan"}, {"../a_plan"}, {"a.b_plan"}, {"a b_plan"}}) {
        bool rejected = false;
        try { validate_minimax_h3_plan_sections(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Plan section lists reject empty, duplicate or unsafe names");
    }
    MiniMaxH3AdalnCoverage coverage{};
    for (int segment = 0; segment < 2; ++segment) {
        for (int layer = segment * 25; layer < (segment + 1) * 25; ++layer)
            check(claim_minimax_h3_adaln_output(coverage, "block_modulation_" + std::to_string(layer),
                                                segment == 0) == layer,
                  "AdaLN segments claim globally named outputs exactly once");
        if (segment == 0)
            check(claim_minimax_h3_adaln_output(coverage, "final_modulation") == 50,
                  "Only the first AdaLN segment produces final modulation");
    }
    validate_minimax_h3_adaln_coverage(coverage);
    for (const char* name : {"block_modulation_0", "block_modulation_50", "block_modulation_00",
                            "unknown", "final_modulation"}) {
        bool rejected = false;
        try { (void)claim_minimax_h3_adaln_output(coverage, name); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "AdaLN rejects duplicate and unknown output names");
    }
    for (int missing : {0, 24, 25, 49, 50}) {
        auto incomplete = coverage;
        incomplete[missing] = false;
        bool rejected = false;
        try { validate_minimax_h3_adaln_coverage(incomplete); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "AdaLN rejects incomplete per-step coverage across segments");
    }
    bool rejected = false;
    try {
        MiniMaxH3AdalnCoverage empty{};
        (void)claim_minimax_h3_adaln_output(empty, "final_modulation", false);
    } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "AdaLN rejects final modulation in a later segment");
    for (std::size_t count : {1U, 2U}) {
        std::array<int, 50> visits{};
        for (std::size_t segment = 0; segment < count; ++segment) {
            const auto range = minimax_h3_tail_layer_range(segment, count);
            for (int layer = range[0]; layer < range[1]; ++layer)
                ++visits[layer];
        }
        check(visits[0] == 0, "Tail segments never execute the head block");
        for (int layer = 1; layer < 50; ++layer)
            check(visits[layer] == 1, "Tail segments cover blocks one through 49 once");
    }
    for (const auto [segment, count] : {std::pair<std::size_t, std::size_t>{0, 0}, {0, 3}, {2, 2}}) {
        rejected = false;
        try { (void)minimax_h3_tail_layer_range(segment, count); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Unsupported tail segment indices and counts are rejected");
    }
    config.denoiser_tail_sections.push_back("denoiser_tail_1_plan");
    rejected = false;
    try { validate_minimax_h3_sampler_config(config); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Original residual-cache bundles cannot select direct-hidden split tails");
    config.sampler = MiniMaxH3Sampler::kTurboEuler;
    config.first_block_cache = false;
    config.scheduler_grid_points = 9;
    config.transformer_forwards = 8;
    validate_minimax_h3_sampler_config(config);
    config.adaln_precompute_sections = {"denoiser_tail_1_plan"};
    rejected = false;
    try { validate_minimax_h3_sampler_config(config); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "A plan section cannot be reused across runtime stage roles");
}

void test_turbo_dual_clock_bfloat16_euler() {
    const auto video = trtmc::make_minimax_h3_turbo_schedule(8, 12.0F);
    const auto audio = trtmc::make_minimax_h3_turbo_schedule(8, 3.0F);
    const std::array<double, 9> expected_video =
        {1.0, 84.0/85.0, 36.0/37.0, 20.0/21.0, 12.0/13.0, 36.0/41.0, 0.8, 12.0/19.0, 0.0};
    const std::array<double, 9> expected_audio =
        {1.0, 21.0/22.0, 0.9, 5.0/6.0, 0.75, 9.0/14.0, 0.5, 0.3, 0.0};
    check(video.sigmas.size() == 9 && video.timesteps.size() == 8 &&
              audio.sigmas.size() == 9 && audio.timesteps.size() == 8,
          "Turbo eight forwards use nine sigma endpoints for both clocks");
    for (std::size_t index = 0; index < 9; ++index) {
        check(video.sigmas[index] == static_cast<float>(expected_video[index]) &&
                  audio.sigmas[index] == static_cast<float>(expected_audio[index]),
              "Turbo dual clock sigma grid matches authored double-precision schedule");
        if (index < 8) {
            check_near(video.timesteps[index], static_cast<float>(1.0 - expected_video[index]),
                       1.0e-7F, "Turbo video timestep uses the data-ward clock");
            check_near(audio.timesteps[index], static_cast<float>(1.0 - expected_audio[index]),
                       1.0e-7F, "Turbo audio timestep uses its independent data-ward clock");
        }
    }
    check_near(audio.slopes.front(), 4.0F, 0.0F, "Turbo initial audio clock slope is four");
    check_near(audio.slopes.back(), 0.9025F, 0.0F, "Turbo final audio clock slope is retained");
    check(trtmc::minimax_h3_round_bfloat16(1.0F + 1.0F/256.0F) == 1.0F &&
              trtmc::minimax_h3_round_bfloat16(1.0F + 3.0F/256.0F) == 1.0F + 2.0F/128.0F,
          "Turbo BF16 conversion rounds ties to even");

    // Recorded from the author's eager expression with CPU PyTorch BF16
    // tensors; this checks arithmetic boundaries, not a CUDA engine oracle.
    const std::vector<float> initial = {1.0F, -2.0F, 0.003921F, 123.4F, -0.75F, 9.99F};
    const std::vector<float> velocity = {0.5011F, 0.2499F, -2.71828F, 7.003F, 1.003F, 12.73F};
    const std::vector<float> expected =
        {1.078125F, -1.9609375F, -0.404296875F, 124.5F, -0.6015625F, 11.9375F};
    for (bool is_audio : {false, true}) {
        auto result = initial;
        trtmc::minimax_h3_turbo_scheduler_step(result.data(), velocity.data(), result.size(),
                                              -0.15F, 0.9025F, is_audio);
        check(result == expected, "Turbo eager BF16 multiply/add matches the recorded oracle");
    }
    float audio_value = 0.0F;
    float video_value = 0.0F;
    const float distinguishing_velocity = -9.5F;
    trtmc::minimax_h3_turbo_scheduler_step(&audio_value, &distinguishing_velocity, 1,
                                          -0.017123456F, 1.927413F, true);
    trtmc::minimax_h3_turbo_scheduler_step(&video_value, &distinguishing_velocity, 1,
                                          -0.017123456F);
    check(audio_value == -0.1611328125F && video_value == -0.1630859375F,
          "Turbo audio retains BF16 slope multiply/divide rather than cancelling them");
}

void test_turbo_geometry_and_audio_packing() {
    const auto geometry = trtmc::make_minimax_h3_geometry(362, 736, 1280, true);
    check(geometry.turbo_profile && geometry.video_latent_frames == 107 &&
              geometry.audio_latent_frames == 603 && geometry.audio_rows == 1206 &&
              geometry.video_rows == 98440,
          "Turbo 362-frame reference geometry keeps all audio and video latents");
    check(trtmc::align_minimax_h3_num_frames(360, true) == 362,
          "Turbo aligns a 360-frame request to its native 362-frame output");
    check(!trtmc::is_minimax_h3_native_canvas(736, 1280) &&
              trtmc::is_minimax_h3_native_canvas(736, 1280, true),
          "Turbo canvas does not silently widen the original public profile");
    for (bool turbo : {false, true}) {
        bool rejected = false;
        try { (void)trtmc::make_minimax_h3_geometry(turbo ? 379 : 362, 768, 1344, turbo); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Original and Turbo profiles retain their own frame maxima");
    }
    const auto largest = trtmc::make_minimax_h3_geometry(362, 576, 1856, true);
    check(largest.video_rows == 111708 &&
              trtmc::minimax_h3_cache_tensor_bytes(2641, largest) ==
                  static_cast<std::size_t>(115555) * 5376U * sizeof(uint16_t),
          "Turbo dynamic prompt and media rows fit the widened engine ABI");
    std::vector<float> source(32 * 2 * 3);
    for (std::size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<float>(index);
    const auto packed = trtmc::pack_minimax_h3_turbo_audio_noise(source, 3);
    for (std::size_t row = 0; row < 6; ++row)
        for (std::size_t channel = 0; channel < 32; ++channel)
            check(packed[row * 32 + channel] == source[channel * 6 + row],
                  "Turbo audio noise preserves authored [32,2,T] generator ordering");
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

void test_turbo_fl2va_geometry() {
    for (const auto canvas : {std::array<int32_t, 2>{768, 1344}, {736, 1280}, {1856, 576}}) {
        for (int32_t frames : {124, 345, 362}) {
            const auto base = trtmc::make_minimax_h3_geometry(frames, canvas[0], canvas[1], true);
            for (int32_t count : {1, 2}) {
                const auto geometry = trtmc::make_minimax_h3_fl2va_geometry(base, count);
                check(geometry.turbo_profile && geometry.output_frames == frames &&
                          geometry.target_video_rows == base.target_video_rows &&
                          geometry.video_rows == base.target_video_rows + geometry.condition_video_rows &&
                          geometry.video_rows <= 113796,
                      "Turbo FL2VA preserves its dynamic canvas and 362-frame envelope");
                const std::vector<int32_t> tags(2641, 1);
                const auto anchors = count == 1 ? std::vector<int32_t>{frames - 1}
                                                : std::vector<int32_t>{0, frames - 1};
                const auto metadata = trtmc::make_minimax_h3_fl2va_denoiser_metadata(
                    tags, anchors, geometry);
                const auto condition_begin = tags.size() + geometry.audio_rows;
                check(metadata.positions.size() ==
                          (tags.size() + geometry.audio_rows + geometry.video_rows) * 3 &&
                          metadata.positions.size() <= 117643U * 3 &&
                          metadata.timestep_indices[condition_begin] == 2 &&
                          metadata.adaln_indices[condition_begin] == 6 &&
                          metadata.timestep_indices[condition_begin + geometry.condition_video_rows] == 0,
                      "Turbo FL2VA keeps fixed conditions and generated rows on separate clocks");
            }
        }
    }
    std::vector<float> packed{9.0F, 8.0F, 1.0F, 2.0F};
    const std::vector<float> velocity{100.0F, 200.0F, 0.5F, 0.25F};
    trtmc::minimax_h3_turbo_scheduler_step(packed.data() + 2, velocity.data() + 2, 2, -0.5F);
    check(packed[0] == 9.0F && packed[1] == 8.0F && packed[2] == 1.25F && packed[3] == 2.125F,
          "Turbo Euler updates the generated suffix without changing keyframe rows");
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
    test_segmented_plan_contracts();
    test_pinned_schedules();
    test_first_block_cache_tail_schedule();
    test_first_block_cache_request_capacity();
    test_data_ward_euler_sign();
    test_turbo_sampler_contract();
    test_turbo_dual_clock_bfloat16_euler();
    test_turbo_geometry_and_audio_packing();
    test_variable_text_position_layout();
    test_prompt_token_profile_boundaries();
    test_denoiser_optimization_profile_selection();
    test_public_video_geometry();
    test_explicit_super_resolution_generation_contract();
    test_public_canvas_resolver_and_vae_tiles();
    test_variable_duration_position_layout();
    test_fl2va_full_public_geometry_and_rotary_contract();
    test_turbo_fl2va_geometry();
    test_audio_latent_unpack_and_denormalize();
    test_audio_decoder_channel_duplication();
    return failures == 0 ? 0 : 1;
}
