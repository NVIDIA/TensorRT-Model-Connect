/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/pipeline.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
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

    const auto fast_video = trtmc::make_minimax_h3_schedule(5, 12.0F);
    const auto fast_audio = trtmc::make_minimax_h3_schedule(5, 3.0F);
    check(fast_video.sigmas.size() == 5 && fast_video.timesteps.size() == 4,
          "FastH3 uses exactly four transformer forwards");
    check(fast_audio.sigmas.size() == 5 && fast_audio.timesteps.size() == 4,
          "FastH3 audio uses the same four-forward grid");
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
    check(trtmc::should_compute_minimax_h3_tail(
              1, forwards, std::numeric_limits<float>::quiet_NaN(), threshold),
          "H3 FirstBlockCache computes an interior tail for a non-finite metric");
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
    for (const auto [tokens, maximum] :
         {std::pair<std::size_t, int32_t>{537, 537}, {2641, 2641}}) {
        try {
            trtmc::validate_minimax_h3_prompt_token_count(tokens, maximum);
        } catch (...) {
            check(false, "H3 prompt accepts the declared profile endpoint");
        }
    }

    for (const auto [tokens, maximum] :
         {std::pair<std::size_t, int32_t>{538, 537}, {2642, 2641}}) {
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
    const auto five_seconds = trtmc::make_minimax_h3_geometry(124, 768, 1344);
    check(trtmc::select_minimax_h3_denoiser_profile(1, 537, five_seconds) == 0,
          "H3 legacy single-profile denoiser always selects profile zero");
    check(trtmc::select_minimax_h3_denoiser_profile(2, 537, five_seconds) == 0,
          "H3 exact qualified five-second request selects the static profile");
    check(trtmc::select_minimax_h3_denoiser_profile(2, 536, five_seconds) == 1,
          "H3 non-reference prompt length selects the public dynamic profile");

    const auto fifteen_seconds = trtmc::make_minimax_h3_geometry(345, 768, 1344);
    check(trtmc::select_minimax_h3_denoiser_profile(2, 537, fifteen_seconds) == 1,
          "H3 fifteen-second request selects the public dynamic profile");

    const auto fl2va = trtmc::make_minimax_h3_fl2va_geometry(five_seconds, 1);
    check(trtmc::select_minimax_h3_denoiser_profile(2, 537, fl2va) == 1,
          "H3 FL2VA request selects the public dynamic profile");

    for (const int32_t profile_count : {0, 3}) {
        bool rejected = false;
        try {
            (void)trtmc::select_minimax_h3_denoiser_profile(profile_count, 537, five_seconds);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "H3 denoiser rejects an unsupported optimization-profile count");
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
        check(short_geometry.video_rows == 18870 && short_geometry.vae_tile_count == 15 &&
                  short_geometry.vsa_video_tiles == 400,
              "H3 124f profile accepts the documented 960x544 explicit canvas");
        const auto long_geometry = trtmc::make_minimax_h3_geometry(345, canvas[0], canvas[1]);
        check(long_geometry.video_rows == 52020 && long_geometry.vae_tile_count == 15 &&
                  long_geometry.vsa_video_tiles == 1040,
              "H3 345f profile accepts the documented 960x544 explicit canvas");
    }

    const auto max_rows = trtmc::make_minimax_h3_geometry(345, 576, 1856);
    check(max_rows.video_rows == 106488,
          "H3 dynamic row profile covers the largest rounded public canvas");
    const auto max_vsa = trtmc::make_minimax_h3_geometry(345, 544, 1952);
    check(max_vsa.vsa_video_tiles == 2080 && max_vsa.vsa_top_video_tiles == 208,
          "H3 metadata exposes the worst continuous-ratio VSA extent");

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
          "H3 continuous aspect resolver preserves the VSA worst-case canvas");
    check(trtmc::make_minimax_h3_geometry(345, continuous_worst.height, continuous_worst.width)
                  .vsa_video_tiles == 2080,
          "H3 continuous aspect metadata retains the 2,080 VSA tile maximum");

    const auto default_tiles = trtmc::make_minimax_h3_vae_tile_layout(768, 1344);
    check(default_tiles.y_starts == std::vector<int32_t>({0, 160, 336, 512}) &&
              default_tiles.y_overlaps == std::vector<int32_t>({96, 80, 80}),
          "H3 dynamic VAE tiler exactly preserves qualified 768p vertical tiles");
    check(default_tiles.x_starts == std::vector<int32_t>({0, 176, 352, 528, 704, 896, 1088}) &&
              default_tiles.x_overlaps == std::vector<int32_t>({80, 80, 80, 80, 64, 64}),
          "H3 dynamic VAE tiler exactly preserves qualified 768p horizontal tiles");

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
                    check(geometry.video_rows <= 108576 && geometry.vsa_video_tiles <= 2080,
                          "H3 FL2VA stays inside frozen video/VSA maxima");
                    const int32_t text_rows = keyframes == 1 ? 600 : 1200;
                    std::vector<int32_t> tags(static_cast<std::size_t>(text_rows), 1);
                    std::vector<int32_t> anchors = keyframes == 1
                                                       ? std::vector<int32_t>{frames - 1}
                                                       : std::vector<int32_t>{0, frames - 1};
                    const auto metadata = trtmc::make_minimax_h3_fl2va_denoiser_metadata(
                        tags, anchors, geometry, true);
                    const int32_t sequence_rows =
                        text_rows + geometry.audio_rows + geometry.video_rows;
                    check(sequence_rows <= 112367 &&
                              metadata.positions.size() ==
                                  static_cast<std::size_t>(sequence_rows) * 3 &&
                              metadata.vsa.video_valid_sizes.size() ==
                                  static_cast<std::size_t>(geometry.vsa_video_tiles),
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
    check(public_canvases == 97,
          "H3 FL2VA validates 95 resolver canvases plus both 960x544 orientations");
}

void check_vsa_metadata(const trtmc::MiniMaxH3VsaMetadata& metadata, int32_t sequence_rows,
                        int32_t prefix_tiles, int32_t video_tiles) {
    check(metadata.packed_row_to_tile_slot.size() == static_cast<std::size_t>(sequence_rows),
          "H3 VSA row map follows the live packed sequence");
    check(metadata.prefix_valid_sizes.size() == static_cast<std::size_t>(prefix_tiles),
          "H3 VSA prefix valid sizes follow the segment-pure tiles");
    check(metadata.video_valid_sizes.size() == static_cast<std::size_t>(video_tiles),
          "H3 VSA video valid sizes follow the 3D tiles");
    check(
        std::accumulate(metadata.prefix_valid_sizes.begin(), metadata.prefix_valid_sizes.end(), 0) +
                std::accumulate(metadata.video_valid_sizes.begin(),
                                metadata.video_valid_sizes.end(), 0) ==
            sequence_rows,
        "H3 VSA valid sizes cover every natural packed row");
    std::vector<int32_t> visits(static_cast<std::size_t>(prefix_tiles + video_tiles) * 64, 0);
    bool unique_and_invertible = metadata.tiled_slot_to_packed_row.size() == visits.size();
    for (std::size_t packed = 0; packed < metadata.packed_row_to_tile_slot.size(); ++packed) {
        const int32_t slot = metadata.packed_row_to_tile_slot[packed];
        if (slot < 0 || slot >= static_cast<int32_t>(visits.size()) || visits[slot]++ != 0)
            unique_and_invertible = false;
        else if (metadata.tiled_slot_to_packed_row[static_cast<std::size_t>(slot)] !=
                 static_cast<int32_t>(packed))
            unique_and_invertible = false;
    }
    check(unique_and_invertible,
          "H3 VSA row map is a one-to-one invertible scatter into padded tile slots");
}

void test_native_vsa_runtime_metadata() {
    {
        const auto geometry = trtmc::make_minimax_h3_geometry(124, 544, 960);
        const auto metadata = trtmc::make_minimax_h3_denoiser_metadata(1, geometry, true);
        check_vsa_metadata(metadata.vsa, 19285, 8, 400);
        check(metadata.positions.size() == static_cast<std::size_t>(19285) * 3 &&
                  metadata.adaln_indices.size() == 19285 &&
                  metadata.timestep_indices.size() == 19285,
              "H3 documented 960x544 canvas binds the new packed-row minimum");
    }
    {
        const auto geometry = trtmc::make_minimax_h3_geometry(124, 768, 768);
        const auto metadata = trtmc::make_minimax_h3_denoiser_metadata(1, geometry, true);
        check_vsa_metadata(metadata.vsa, 21727, 8, 360);
        check(metadata.positions.size() == static_cast<std::size_t>(21727) * 3 &&
                  metadata.adaln_indices.size() == 21727 &&
                  metadata.timestep_indices.size() == 21727,
              "H3 native VSA min profile binds dynamic RoPE and AdaLN indices");
        check(metadata.vsa.prefix_valid_sizes ==
                  std::vector<int32_t>({1, 64, 64, 64, 64, 64, 64, 30}),
              "H3 native VSA keeps text and audio in separate prefix tiles");
    }
    {
        const auto geometry = trtmc::make_minimax_h3_geometry(124, 768, 1344);
        const auto metadata = trtmc::make_minimax_h3_denoiser_metadata(128, geometry, true);
        check_vsa_metadata(metadata.vsa, 37838, 9, 660);
        const int32_t video_begin = 128 + geometry.audio_rows;
        check(metadata.vsa.packed_row_to_tile_slot[static_cast<std::size_t>(video_begin)] == 9 * 64,
              "H3 native VSA video tiles begin after the segment-pure prefix");
        check(metadata.vsa.packed_row_to_tile_slot[static_cast<std::size_t>(video_begin + 42)] ==
                  9 * 64 + 4,
              "H3 native VSA preserves video H/W raster order inside a tile");
    }
    {
        const auto geometry = trtmc::make_minimax_h3_geometry(345, 544, 1952);
        const auto metadata = trtmc::make_minimax_h3_denoiser_metadata(537, geometry, true);
        check_vsa_metadata(metadata.vsa, 537 + geometry.audio_rows + geometry.video_rows, 27, 2080);
        check(metadata.vsa.video_valid_sizes.back() == 2,
              "H3 native VSA clips the worst-aspect temporal/spatial edge tile");
    }
    {
        const auto geometry = trtmc::make_minimax_h3_geometry(345, 544, 1952);
        const auto metadata = trtmc::make_minimax_h3_denoiser_metadata(2641, geometry, true);
        check_vsa_metadata(metadata.vsa, 2641 + geometry.audio_rows + geometry.video_rows, 60,
                           2080);
        check(metadata.vsa.prefix_valid_sizes.back() == 62,
              "H3 unified FL2VA envelope keeps the final audio prefix tile segment-pure");
    }
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
    test_pinned_schedules();
    test_first_block_cache_tail_schedule();
    test_data_ward_euler_sign();
    test_variable_text_position_layout();
    test_prompt_token_profile_boundaries();
    test_denoiser_optimization_profile_selection();
    test_public_video_geometry();
    test_public_canvas_resolver_and_vae_tiles();
    test_variable_duration_position_layout();
    test_fl2va_full_public_geometry_and_rotary_contract();
    test_native_vsa_runtime_metadata();
    test_audio_latent_unpack_and_denormalize();
    test_audio_decoder_channel_duplication();
    return failures == 0 ? 0 : 1;
}
