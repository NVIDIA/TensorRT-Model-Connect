/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/vae_encoder_tiles.h"

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

void test_canonical_axis_plans() {
    const auto default_canvas = trtmc::make_minimax_h3_vae_spatial_tile_plan(768, 1344);
    check(default_canvas.rows.size() == 4 && default_canvas.columns.size() == 7,
          "H3 default canvas uses the canonical 4x7 VAE tiles");
    check(default_canvas.rows[0].latent_crop_after == 6 &&
              default_canvas.rows[1].latent_crop_after == 5 &&
              default_canvas.rows[2].latent_crop_after == 5,
          "H3 default vertical latent overlaps are 6,5,5");
    check(default_canvas.columns[0].latent_crop_after == 5 &&
              default_canvas.columns[4].latent_crop_after == 4,
          "H3 default horizontal overlap distribution matches Diffusers");

    const auto image = trtmc::make_minimax_h3_vae_spatial_tile_plan(2048, 8192);
    check(image.rows.size() == 11 && image.columns.size() == 43,
          "H3 4:1 reference image exposes every static spatial tile");
    check(image.rows.back().stitch_start + image.rows.back().stitch_length == 128 &&
              image.columns.back().stitch_start + image.columns.back().stitch_length == 512,
          "H3 stitched image plan covers the exact latent canvas");
}

void test_temporal_plan_pads_then_drops_once() {
    const auto image = trtmc::make_minimax_h3_vae_temporal_chunk_plan(1);
    check(image.chunks.size() == 1 && image.output_moment_frames == 1 && image.token_drop == 0,
          "H3 still image uses the isolated T=1 path");
    const auto video = trtmc::make_minimax_h3_vae_temporal_chunk_plan(39);
    check(video.chunks.size() == 3 && video.raw_moment_frames == 15 &&
              video.output_moment_frames == 12,
          "H3 39-frame video concatenates three raw chunks then drops three moments");
    check(video.chunks.back().valid_input_frames == 5 &&
              video.chunks.back().repeated_tail_frames == 12 &&
              video.chunks.back().repeat_source_frame == 38,
          "H3 partial temporal chunk repeats the last source frame");
}

void test_horizontal_stitch_uses_unstitched_neighbor() {
    const auto plan = trtmc::make_minimax_h3_vae_spatial_tile_plan(256, 448);
    const std::size_t tile_elements = 48U * 16U * 16U;
    std::vector<std::vector<float>> tiles(2);
    tiles[0].assign(tile_elements, 10.0F);
    tiles[1].assign(tile_elements, 20.0F);
    const auto stitched = trtmc::minimax_h3_stitch_vae_encoder_tiles(tiles, plan, 1);
    check(stitched.size() == 48U * 16U * 28U,
          "H3 stitched width removes exactly one four-latent overlap");
    const auto at = [&](int32_t x) { return stitched[static_cast<std::size_t>(x)]; };
    check(at(11) == 10.0F && at(12) == 10.0F && at(13) == 12.5F && at(14) == 15.0F &&
              at(15) == 17.5F && at(16) == 20.0F,
          "H3 overlap weights are previous=1-i/N and current=i/N");
}

void test_temporal_assembly_is_channel_major() {
    const auto plan = trtmc::make_minimax_h3_vae_temporal_chunk_plan(18);
    std::vector<std::vector<float>> chunks(2, std::vector<float>(48U * 5U));
    for (int32_t channel = 0; channel < 48; ++channel) {
        for (int32_t frame = 0; frame < 5; ++frame) {
            chunks[0][static_cast<std::size_t>(channel) * 5U + frame] =
                static_cast<float>(channel * 100 + frame);
            chunks[1][static_cast<std::size_t>(channel) * 5U + frame] =
                static_cast<float>(channel * 100 + 10 + frame);
        }
    }
    const auto moments = trtmc::minimax_h3_assemble_vae_temporal_moments(chunks, 1, 1, plan);
    check(moments.size() == 48U * 7U, "H3 temporal assembly drops three of ten raw frames");
    check(moments[6] == 11.0F && moments[7] == 100.0F,
          "H3 temporal assembly concatenates inside each channel before the global drop");
}

} // namespace

int main() {
    test_canonical_axis_plans();
    test_temporal_plan_pads_then_drops_once();
    test_horizontal_stitch_uses_unstitched_neighbor();
    test_temporal_assembly_is_channel_major();
    return failures == 0 ? 0 : 1;
}
