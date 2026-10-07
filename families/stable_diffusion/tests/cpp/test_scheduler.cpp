/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_diffusion/runtime/scheduler.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::vector<float> schedule(int n) {
    // A stand-in alphas_cumprod: strictly decreasing from ~1 towards 0, which
    // is the only property the DDIM update depends on.
    std::vector<float> alphas(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        alphas[static_cast<std::size_t>(i)] =
            1.0F - 0.9F * static_cast<float>(i) / static_cast<float>(n);
    return alphas;
}

void test_timesteps_descend_and_count() {
    trtmc::stable_diffusion::DdimScheduler s(schedule(1000), 1000, 1);
    const auto steps = s.timesteps(10);
    check(steps.size() == 10U, "one timestep per requested step");
    for (std::size_t i = 1; i < steps.size(); ++i)
        check(steps[i] < steps[i - 1], "timesteps descend");
    check(steps.back() == 1, "the walk ends at the steps_offset");
}

void test_step_count_is_validated() {
    trtmc::stable_diffusion::DdimScheduler s(schedule(1000), 1000, 1);
    bool threw = false;
    try {
        s.timesteps(0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "zero steps is rejected");
    threw = false;
    try {
        s.timesteps(1001);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "more steps than the training schedule is rejected");
}

void test_a_mismatched_schedule_is_rejected() {
    bool threw = false;
    try {
        trtmc::stable_diffusion::DdimScheduler s(schedule(10), 1000, 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "alphas_cumprod must cover the training schedule");
}

void test_zero_noise_leaves_a_consistent_sample() {
    trtmc::stable_diffusion::DdimScheduler s(schedule(1000), 1000, 1);
    std::vector<float> latents{1.0F, -2.0F, 0.5F};
    const std::vector<float> noise(3, 0.0F);
    const auto before = latents;
    s.step(noise.data(), latents.data(), latents.size(), 500, 400);
    // With no predicted noise the update is a pure rescale, so signs hold and
    // nothing becomes non-finite.
    for (std::size_t i = 0; i < latents.size(); ++i) {
        check(std::isfinite(latents[i]), "the update stays finite");
        check((latents[i] >= 0.0F) == (before[i] >= 0.0F), "the update preserves sign");
    }
}

void test_the_final_step_targets_alpha_one() {
    trtmc::stable_diffusion::DdimScheduler s(schedule(1000), 1000, 1);
    std::vector<float> latents{0.3F};
    const std::vector<float> noise{0.0F};
    s.step(noise.data(), latents.data(), 1, 500, -1);
    check(std::isfinite(latents[0]), "the last step is finite");
}

} // namespace

int main() {
    test_timesteps_descend_and_count();
    test_step_count_is_validated();
    test_a_mismatched_schedule_is_rejected();
    test_zero_noise_leaves_a_consistent_sample();
    test_the_final_step_targets_alpha_one();
    if (failures != 0) {
        std::fprintf(stderr, "%d stable_diffusion scheduler check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("stable_diffusion scheduler checks passed\n");
    return EXIT_SUCCESS;
}
