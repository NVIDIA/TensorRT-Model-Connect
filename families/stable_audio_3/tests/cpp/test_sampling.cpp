/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host sampling arithmetic of the Stable Audio 3 runtime. The golden values are
// the ones tests/test_sampling.py asserts for the NumPy twin.

#include "families/stable_audio_3/runtime/sampling.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace sa3 = trtmc::stable_audio_3;

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", what);
        ++failures;
    }
}

void expect_close(const std::vector<float>& actual, const std::vector<float>& expected,
                  float tolerance, const char* what) {
    bool same = actual.size() == expected.size();
    for (std::size_t index = 0; same && index < actual.size(); ++index)
        same = std::fabs(actual[index] - expected[index]) <=
               tolerance * (1.0F + std::fabs(expected[index]));
    expect(same, what);
}

void test_noise() {
    std::vector<float> noise(7);
    sa3::gaussian_noise(42, 0, noise.data(), noise.size());
    expect_close(noise,
                 {-0.318013877F, -1.34076893F, 0.258929193F, 0.470199823F, 1.16252971F,
                  -0.672291458F, -0.419299811F},
                 1e-6F, "noise draw 0");
    sa3::gaussian_noise(42, 3, noise.data(), noise.size());
    expect_close(noise,
                 {1.17811048F, -0.363997906F, 1.00697529F, 1.75008237F, -0.573026538F,
                  -0.352358818F, 1.22086978F},
                 1e-6F, "noise draw 3");
    expect(sa3::splitmix64(0) == 0xE220A8397B1DCDAFULL, "splitmix64 reference output");
}

void test_schedule() {
    const auto times = sa3::schedule(8, 108, sa3::LogSnrShift{});
    expect_close(times,
                 {1.0F, 0.994375587F, 0.984480262F, 0.957912266F, 0.890903175F, 0.745546579F,
                  0.512497365F, 0.273885012F, 0.0F},
                 2e-7F, "LogSNR schedule");
    std::vector<float> half;
    for (const auto value : times)
        half.push_back(sa3::round_to_half(value));
    expect_close(half,
                 {1.0F, 0.994140625F, 0.984375F, 0.958007812F, 0.891113281F, 0.745605469F,
                  0.512695312F, 0.273925781F, 0.0F},
                 0.0F, "schedule rounded to half");
    bool rejected = false;
    try {
        (void)sa3::schedule(0, 1, sa3::LogSnrShift{});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "zero steps rejected");
}

void test_round_to_half() {
    expect(sa3::round_to_half(0.1F) == 0.0999755859375F, "0.1 rounds to half");
    expect(sa3::round_to_half(65519.0F) == 65504.0F, "largest finite half");
    expect(std::isinf(sa3::round_to_half(65520.0F)), "half overflow");
    expect(sa3::round_to_half(-2.0004883F) == -2.0F, "negative rounding");
    expect(sa3::round_to_half(1e-6F) == 1.01327896e-06F, "half subnormal");
}

void test_render_plan() {
    const sa3::RenderLimits limits{};
    struct Case {
        double seconds;
        int latents;
        long long effective;
        int valid;
        long long samples;
    };
    const Case cases[] = {
        {0.5, 70, 6, 70, 22050},
        {10.0, 174, 108, 172, 441000},
        {47.3, 574, 510, 574, 2085929},
        {114.0, 1292, 1228, 1292, 5027400},
        {120.0, 1292, 1292, 1292, 5292000},
        {300.0, 1292, 3230, 1292, 5292032},
        {3.0e8, 1292, 3229980469LL, 1292, 5292032},
    };
    for (const auto& item : cases) {
        const auto plan = sa3::plan_render(item.seconds, limits);
        expect(plan.latents == item.latents && plan.effective_latents == item.effective &&
                   plan.valid_latents == item.valid && plan.output_samples == item.samples,
               "render plan");
    }
    for (const double seconds : {0.0, 1.0e9 * 1.0000001}) {
        bool rejected = false;
        try {
            (void)sa3::plan_render(seconds, limits);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "duration outside (0, 1e9] rejected");
    }
}

} // namespace

int main() {
    test_noise();
    test_schedule();
    test_round_to_half();
    test_render_plan();
    if (failures != 0)
        return EXIT_FAILURE;
    std::printf("stable_audio_3 sampling: all checks passed\n");
    return EXIT_SUCCESS;
}
