/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The expected values were produced by diffusers 0.39.0:
// FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0).set_timesteps(n)
// and get_timestep_embedding(t, 256, flip_sin_to_cos=True, downscale_freq_shift=0).

#include "families/stable_diffusion_3/runtime/sampling.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace sd3 = trtmc::stable_diffusion_3;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void check_bits(const std::vector<float>& actual, const std::vector<std::uint32_t>& expected,
                const char* what) {
    bool same = actual.size() == expected.size();
    for (std::size_t i = 0; same && i < actual.size(); ++i)
        same = bits(actual[i]) == expected[i];
    check(same, what);
}

void test_schedules_match_diffusers_bit_for_bit() {
    const auto four = sd3::flow_match_schedule(4, 1000, 3.0);
    check_bits(four.sigmas, {0x3f800000U, 0x3f5b91b9U, 0x3f1a268aU, 0x3c124925U, 0x00000000U},
               "4-step sigmas");
    check_bits(four.timesteps, {0x447a0000U, 0x44566c4fU, 0x441689a3U, 0x410edb6eU},
               "4-step timesteps");

    const auto twenty_eight = sd3::flow_match_schedule(28, 1000, 3.0);
    check_bits(twenty_eight.sigmas,
               {0x3f800000U, 0x3f7cc4faU, 0x3f795f20U, 0x3f75cb09U, 0x3f7204eaU, 0x3f6e088cU,
                0x3f69d13dU, 0x3f6559b9U, 0x3f609c1aU, 0x3f5b91b9U, 0x3f563310U, 0x3f507797U,
                0x3f4a5593U, 0x3f43c1e1U, 0x3f3cafb1U, 0x3f351034U, 0x3f2cd232U, 0x3f23e18aU,
                0x3f1a268aU, 0x3f0f851fU, 0x3f03dbc4U, 0x3eee043bU, 0x3ed18e4eU, 0x3eb1dd73U,
                0x3e8e5c69U, 0x3e4c9f72U, 0x3de3228bU, 0x3c124925U, 0x00000000U},
               "28-step sigmas");
    check_bits(twenty_eight.timesteps,
               {0x447a0000U, 0x4476d85cU, 0x447386e5U, 0x44700847U, 0x446c58cdU, 0x44687459U,
                0x44645656U, 0x445ff99fU, 0x445b5871U, 0x44566c4fU, 0x44512ddeU, 0x444b94c9U,
                0x44459792U, 0x443f2b56U, 0x44384393U, 0x4430d1d3U, 0x4428c545U, 0x44200a41U,
                0x441689a3U, 0x440c2800U, 0x4400c49dU, 0x43e87022U, 0x43cca4f8U, 0x43adb242U,
                0x438b063fU, 0x4347d3b5U, 0x42ddcfbcU, 0x410edb6eU},
               "28-step timesteps");

    const auto one = sd3::flow_match_schedule(1, 1000, 3.0);
    check(one.sigmas.size() == 2U && one.sigmas[0] == 1.0F && one.sigmas[1] == 0.0F,
          "a one-step schedule walks from 1 straight to 0");
    check(one.timesteps.size() == 1U && one.timesteps[0] == 1000.0F,
          "a one-step schedule evaluates at t = 1000");
}

void test_invalid_schedules_are_rejected() {
    bool threw = false;
    try {
        (void)sd3::flow_match_schedule(0, 1000, 3.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "zero steps are rejected");
}

void test_timestep_projection_matches_diffusers() {
    struct Case {
        float timestep;
        std::vector<std::pair<int, double>> expected;
    };
    const Case cases[] = {
        {1000.0F,
         {{0, 0.5623790621757507},
          {1, 0.7896266579627991},
          {63, -0.2469172477722168},
          {127, 0.994231641292572},
          {128, 0.8268795609474182},
          {129, 0.6135876178741455},
          {191, -0.9690365791320801},
          {255, 0.10725405812263489}}},
        {8.928571701049805F,
         {{0, -0.879395067691803},
          {1, -0.43919098377227783},
          {63, 0.995400607585907},
          {127, 0.9999995231628418},
          {128, 0.47609278559684753},
          {129, 0.8983937501907349},
          {191, 0.09579997509717941},
          {255, 0.0009594709263183177}}},
    };
    std::vector<float> projection(256);
    for (const auto& item : cases) {
        sd3::timestep_projection(item.timestep, 256, projection.data());
        for (const auto& [index, value] : item.expected) {
            // libm and torch may differ in the last bit of sin/cos.
            check(std::fabs(projection[static_cast<std::size_t>(index)] - value) < 2e-6,
                  "timestep projection matches get_timestep_embedding");
        }
    }
}

void test_clip_framing_and_pooling() {
    const auto ids = sd3::frame_clip_ids({10, 11, 12}, 49406, 49407, 0, 8);
    check(ids == std::vector<std::int32_t>({49406, 10, 11, 12, 49407, 0, 0, 0}),
          "CLIP ids are bracketed and padded");
    check(sd3::clip_pool_index(ids, 2) == 4, "legacy configs pool at the end-of-text token");

    const auto padded_with_eos = sd3::frame_clip_ids({}, 49406, 49407, 49407, 5);
    check(sd3::clip_pool_index(padded_with_eos, 2) == 1,
          "padding with the end-of-text token keeps the first one");

    const auto truncated = sd3::frame_clip_ids({1, 2, 3, 4, 5, 6, 7}, 49406, 49407, 0, 5);
    check(truncated == std::vector<std::int32_t>({49406, 1, 2, 3, 49407}),
          "long CLIP prompts keep both brackets");
    check(sd3::clip_pool_index(truncated, 49407) == 4, "new configs pool at the first eos id");
}

void test_t5_framing() {
    check(sd3::frame_t5_ids({5, 6}, 1, 0, 5) == std::vector<std::int32_t>({5, 6, 1, 0, 0}),
          "T5 ids end with </s> and are padded");
    check(sd3::frame_t5_ids({5, 6, 7, 8}, 1, 0, 3) == std::vector<std::int32_t>({5, 6, 1}),
          "long T5 prompts keep </s>");
    check(sd3::frame_t5_ids({}, 1, 0, 3) == std::vector<std::int32_t>({1, 0, 0}),
          "the empty prompt is </s> alone");
}

} // namespace

int main() {
    test_schedules_match_diffusers_bit_for_bit();
    test_invalid_schedules_are_rejected();
    test_timestep_projection_matches_diffusers();
    test_clip_framing_and_pooling();
    test_t5_framing();
    if (failures != 0) {
        std::fprintf(stderr, "%d stable_diffusion_3 sampling check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("stable_diffusion_3 sampling checks passed\n");
    return EXIT_SUCCESS;
}
