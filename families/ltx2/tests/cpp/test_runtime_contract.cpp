/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// LTX-2.5 host-side runtime contract: scheduler step, prompt padding, audio interleave, tiled
// VAE ramps, tile latent gather and blend.

#include "families/ltx2/runtime/progress_log.h"
#include "families/ltx2/runtime/runtime_math.h"
#include "families/ltx2/runtime/vae_tiling.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void test_euler_step_matches_flow_match_euler() {
    // diffusers: prev = x + (sigma_next - sigma) * v; the last distilled step lands on x0.
    std::vector<float> x{1.0F, -2.0F, 0.5F};
    const std::vector<float> v{0.5F, 1.0F, -4.0F};
    trtmc::ltx2_euler_step(x, v, 0.421875F, 0.0F);
    check(std::fabs(x[0] - (1.0F - 0.421875F * 0.5F)) < 1e-7F, "euler step value 0");
    check(std::fabs(x[1] - (-2.0F - 0.421875F)) < 1e-7F, "euler step value 1");
    check(std::fabs(x[2] - (0.5F + 0.421875F * 4.0F)) < 1e-7F, "euler step value 2");
    bool threw = false;
    try {
        std::vector<float> short_v{1.0F};
        trtmc::ltx2_euler_step(x, short_v, 1.0F, 0.5F);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "euler step rejects mismatched sizes");
}

void test_two_stage_renoise() {
    // diffusers _create_noised_state: noise_scale * noise + (1 - noise_scale) * latents.
    std::vector<float> x{1.0F, -2.0F};
    trtmc::ltx2_renoise(x, {0.5F, 4.0F}, 0.25F);
    check(x[0] == 0.25F * 0.5F + 0.75F * 1.0F && x[1] == 0.25F * 4.0F + 0.75F * -2.0F,
          "re-noise mixes noise and latents");
    bool threw = false;
    try {
        trtmc::ltx2_renoise(x, {1.0F}, 0.5F);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "re-noise rejects mismatched sizes");
}

void test_prompt_ids_left_pad_and_truncate() {
    std::vector<int32_t> ids;
    std::vector<int32_t> mask;
    trtmc::ltx2_prompt_ids({11, 12, 13}, 6, 0, ids, mask);
    check((ids == std::vector<int32_t>{0, 0, 0, 11, 12, 13}), "left padding");
    check((mask == std::vector<int32_t>{0, 0, 0, 1, 1, 1}), "mask marks tokens");
    trtmc::ltx2_prompt_ids({1, 2, 3, 4, 5}, 3, 0, ids, mask);
    check((ids == std::vector<int32_t>{1, 2, 3}), "right truncation keeps the first tokens");
    check((mask == std::vector<int32_t>{1, 1, 1}), "truncated mask is full");
}

void test_interleave_stereo() {
    const auto out = trtmc::ltx2_interleave({1.0F, 2.0F, 3.0F, -1.0F, -2.0F, -3.0F}, 2);
    check((out == std::vector<float>{1.0F, -1.0F, 2.0F, -2.0F, 3.0F, -3.0F}),
          "planar to interleaved");
}

void test_progress_line_format() {
    const auto line = trtmc::format_ltx2_progress(1, 12.5, "step", "step=2/8 step_ms=3.000");
    check(line == "[ltx-progress] rank=1 t_ms=12.500 event=step step=2/8 step_ms=3.000",
          "progress line keeps the LTX format");
}

uint16_t to_half(float value) {
    // Exact for the small dyadic test values used below.
    const auto bits = [&] {
        uint32_t b;
        std::memcpy(&b, &value, sizeof(b));
        return b;
    }();
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    const int32_t exp = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    if ((bits & 0x7FFFFFFFU) == 0U)
        return static_cast<uint16_t>(sign);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10U) |
                                 ((bits >> 13U) & 0x3FFU));
}

void test_vae_axis_weights() {
    using trtmc::ltx2::vae_axis_weights;
    const auto spatial = vae_axis_weights(6, 2, 2, false);
    check(spatial[0] == 1.0F / 3.0F && spatial[1] == 2.0F / 3.0F && spatial[2] == 1.0F,
          "spatial ramp fades in as k / (r + 1)");
    check(spatial[4] == 1.0F - 1.0F / 3.0F && spatial[5] == 1.0F - 2.0F / 3.0F,
          "spatial ramp fades out as 1 - k / (r + 1)");
    const auto temporal = vae_axis_weights(5, 3, 0, true);
    check(temporal[0] == 0.0F && temporal[1] == 1.0F / 3.0F && temporal[3] == 1.0F,
          "temporal ramp fades in from zero");
    bool threw = false;
    try {
        (void)vae_axis_weights(2, 3, 0, false);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "ramp longer than its tile is rejected");
}

// Two tiles along the width of a 1-frame, 1x6 video, overlapping by 2 pixels.
trtmc::ltx2::VaeTilePlan two_tile_plan() {
    trtmc::ltx2::VaeTilePlan plan;
    plan.tile_latent = {1, 1, 4};
    plan.tile_pixels = {1, 1, 4};
    plan.tiles.push_back({{0, 0, 0}, {0, 0, 0}, {{{0, 0}, {0, 0}, {0, 2}}}, 0});
    plan.tiles.push_back({{0, 0, 2}, {0, 0, 2}, {{{0, 0}, {0, 0}, {2, 0}}}, 1});
    return plan;
}

void test_vae_blend_normalizes_overlaps() {
    const auto plan = two_tile_plan();
    std::vector<uint16_t> left(12, to_half(0.25F));
    std::vector<uint16_t> right(12, to_half(0.75F));
    std::vector<float> out;
    trtmc::ltx2::vae_blend_tiles(plan, {left.data(), right.data()}, 1, 1, 6, out, 1);
    check(out.size() == 18U, "blend output covers the video");
    check(out[0] == 0.25F && out[3 * 5] == 0.75F, "unshared pixels keep their tile");
    const float w = 1.0F / 3.0F;
    const float expected = ((1.0F - w) * 0.25F + w * 0.75F) / ((1.0F - w) + w);
    check(out[3 * 2] == expected, "overlap is the weight-normalized blend");
    std::vector<uint16_t> big(12, to_half(2.0F));
    trtmc::ltx2::vae_blend_tiles(plan, {big.data(), big.data()}, 1, 1, 6, out, 1);
    check(out[3 * 3] == 1.0F, "blended values are clamped to [0, 1]");
}

void test_vae_blend_is_thread_count_invariant() {
    trtmc::ltx2::VaeTilePlan plan;
    plan.tile_latent = {2, 1, 2};
    plan.tile_pixels = {9, 2, 3};
    plan.tiles.push_back({{0, 0, 0}, {0, 0, 0}, {{{0, 0}, {0, 0}, {0, 2}}}, 0});
    plan.tiles.push_back({{0, 0, 1}, {0, 0, 1}, {{{0, 0}, {0, 0}, {2, 0}}}, 1});
    std::vector<uint16_t> a(9 * 2 * 3 * 3);
    std::vector<uint16_t> b(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = to_half(static_cast<float>(i % 7) / 8.0F);
        b[i] = to_half(static_cast<float>(i % 5) / 8.0F);
    }
    std::vector<float> one;
    std::vector<float> many;
    trtmc::ltx2::vae_blend_tiles(plan, {a.data(), b.data()}, 9, 2, 4, one, 1);
    trtmc::ltx2::vae_blend_tiles(plan, {a.data(), b.data()}, 9, 2, 4, many, 4);
    check(one == many, "blend is bit-identical for any thread count");
}

void test_vae_tile_latents_and_validation() {
    // Packed [F=2, H=2, W=3, C=2] latents; tile of 2x1x2 latents at (0, 1, 1).
    std::vector<float> packed(2 * 2 * 3 * 2);
    for (std::size_t i = 0; i < packed.size(); ++i)
        packed[i] = static_cast<float>(i);
    trtmc::ltx2::VaeTilePlan plan;
    plan.tile_latent = {2, 1, 2};
    plan.tile_pixels = {9, 32, 64};
    plan.tiles.push_back({{0, 1, 1}, {0, 32, 32}, {{{0, 0}, {0, 0}, {0, 0}}}, 0});
    std::vector<float> tile;
    trtmc::ltx2::vae_gather_tile_latents(packed, {2, 2, 3}, 2, plan, plan.tiles[0], tile);
    // Tokens (f, h, w) = (0,1,1), (0,1,2), (1,1,1), (1,1,2) -> token ids 4, 5, 10, 11.
    check((tile == std::vector<float>{8, 9, 10, 11, 20, 21, 22, 23}), "tile latent gather");
    trtmc::ltx2::vae_validate_plan(plan, {2, 2, 3}, {9, 64, 96}, 1);
    bool threw = false;
    try {
        trtmc::ltx2::vae_validate_plan(plan, {2, 2, 3}, {9, 64, 64}, 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "tile outside the video is rejected");
    threw = false;
    try {
        plan.tiles[0].rank = 1;
        trtmc::ltx2::vae_validate_plan(plan, {2, 2, 3}, {9, 64, 96}, 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "tile rank outside the world is rejected");
}

} // namespace

int main() {
    test_euler_step_matches_flow_match_euler();
    test_two_stage_renoise();
    test_prompt_ids_left_pad_and_truncate();
    test_interleave_stereo();
    test_progress_line_format();
    test_vae_axis_weights();
    test_vae_blend_normalizes_overlaps();
    test_vae_blend_is_thread_count_invariant();
    test_vae_tile_latents_and_validation();
    if (failures != 0)
        return EXIT_FAILURE;
    std::puts("ltx2 runtime contract: OK");
    return EXIT_SUCCESS;
}
