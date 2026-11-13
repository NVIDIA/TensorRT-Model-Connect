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

// The schedule SDXL actually ships: betas spaced linearly in sqrt-space
// ("scaled_linear") between 0.00085 and 0.012, then alphas_cumprod.
std::vector<float> sdxl_schedule(int n) {
    std::vector<float> alphas(static_cast<std::size_t>(n));
    const double lo = std::sqrt(0.00085);
    const double hi = std::sqrt(0.012);
    double running = 1.0;
    for (int i = 0; i < n; ++i) {
        const double root = lo + (hi - lo) * i / (n - 1);
        running *= 1.0 - root * root;
        alphas[static_cast<std::size_t>(i)] = static_cast<float>(running);
    }
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

// --- Euler-Ancestral, the SDXL branch ---------------------------------------

void test_euler_timesteps_are_trailing_spaced() {
    trtmc::stable_diffusion::EulerAncestralScheduler s(schedule(1000), 1000);
    const auto one = s.timesteps(1);
    check(one.size() == 1U, "one step yields one timestep");
    // Trailing spacing puts the single turbo step on the last training index.
    check(one[0] == 999, "a one-step walk starts at 999");
    const auto four = s.timesteps(4);
    check(four.size() == 4U, "four steps yield four timesteps");
    check(four[0] == 999, "the four-step walk also starts at 999");
    for (std::size_t i = 1; i < four.size(); ++i)
        check(four[i] < four[i - 1], "the walk descends");
}

void test_euler_step_count_is_validated() {
    trtmc::stable_diffusion::EulerAncestralScheduler s(schedule(1000), 1000);
    bool threw = false;
    try {
        (void)s.timesteps(0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a non-positive step count is rejected");
}

void test_sigma_rises_with_the_timestep() {
    trtmc::stable_diffusion::EulerAncestralScheduler s(schedule(1000), 1000);
    check(s.sigma_at(999) > s.sigma_at(500), "later timesteps are noisier");
    check(s.sigma_at(500) > s.sigma_at(0), "earlier timesteps are cleaner");
}

void test_init_noise_sigma_is_the_largest_in_the_walk() {
    // Measured against the real schedule, not the stand-in: the stand-in
    // bottoms out at alpha 0.1, so its largest sigma is only 3.0 and it cannot
    // show how far from unity the true starting point sits.
    trtmc::stable_diffusion::EulerAncestralScheduler s(sdxl_schedule(1000), 1000);
    const auto walk = s.timesteps(4);
    const double start = s.init_noise_sigma(walk);
    for (const auto timestep : walk)
        check(start >= s.sigma_at(timestep), "no step is noisier than the start");
    check(start == s.sigma_at(walk[0]), "the walk starts at its noisiest point");
    // Omitting this factor is the failure that scored PSNR 7.83 instead of
    // 31.71: the latents enter the UNet an order of magnitude too quiet.
    check(start > 14.0 && start < 15.0, "the starting sigma is about 14.6");
}

void test_scale_input_divides_by_the_noise_level() {
    trtmc::stable_diffusion::EulerAncestralScheduler s(schedule(1000), 1000);
    const std::vector<float> latents{2.0F, -4.0F};
    std::vector<float> scaled(2);
    s.scale_input(latents.data(), scaled.data(), latents.size(), 999);
    const double sigma = s.sigma_at(999);
    const double factor = 1.0 / std::sqrt(sigma * sigma + 1.0);
    for (std::size_t i = 0; i < latents.size(); ++i)
        check(std::fabs(scaled[i] - latents[i] * factor) < 1e-5, "the input is scaled down");
}

void test_the_last_euler_step_is_deterministic() {
    trtmc::stable_diffusion::EulerAncestralScheduler s(schedule(1000), 1000);
    const std::vector<float> noise{0.25F, -0.5F};
    const std::vector<float> loud(2, 1000.0F);
    std::vector<float> with_noise{0.3F, 0.7F};
    std::vector<float> without{0.3F, 0.7F};
    // sigma_next is 0 on the last step, so sigma_up is 0 and the injected
    // gaussian must be discarded no matter how large it is.
    s.step(noise.data(), with_noise.data(), 2, 999, -1, loud.data());
    s.step(noise.data(), without.data(), 2, 999, -1, nullptr);
    for (std::size_t i = 0; i < 2; ++i)
        check(with_noise[i] == without[i], "the final step ignores the gaussian");
}

} // namespace

int main() {
    test_timesteps_descend_and_count();
    test_step_count_is_validated();
    test_a_mismatched_schedule_is_rejected();
    test_zero_noise_leaves_a_consistent_sample();
    test_the_final_step_targets_alpha_one();
    test_euler_timesteps_are_trailing_spaced();
    test_euler_step_count_is_validated();
    test_sigma_rises_with_the_timestep();
    test_init_noise_sigma_is_the_largest_in_the_walk();
    test_scale_input_divides_by_the_noise_level();
    test_the_last_euler_step_is_deterministic();
    if (failures != 0) {
        std::fprintf(stderr, "%d stable_diffusion scheduler check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("stable_diffusion scheduler checks passed\n");
    return EXIT_SUCCESS;
}
