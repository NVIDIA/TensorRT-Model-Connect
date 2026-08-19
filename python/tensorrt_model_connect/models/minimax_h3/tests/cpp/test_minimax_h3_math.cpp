/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline.h"

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

} // namespace

int main() {
    test_pinned_schedules();
    test_data_ward_euler_sign();
    return failures == 0 ? 0 : 1;
}
