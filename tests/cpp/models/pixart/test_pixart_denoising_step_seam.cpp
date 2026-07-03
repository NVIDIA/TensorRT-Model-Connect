/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-PIXART-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-PIXART-01
// Intent:         PixArt-owned denoising step runner updates latents correctly
// Preconditions:  PixArt step runner with mock forward function
// Postconditions: Latents updated after each step, failures propagated, zero steps handled
// =============================================================================

#include "runtime/models/pixart/pixart_denoising_step_seam.h"
#include "runtime/models/pixart/pixart_dpmsolver.h"
#include "runtime/models/pixart/pixart_generation_conditioning.h"
#include "runtime/models/pixart/pixart_generation_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_pixart_step_runner_updates_latents_and_handles_failure() {
    std::vector<float> timesteps = {30.0F, 20.0F};
    std::vector<float> latents = {0.0F, 1.0F, 2.0F};
    std::string error;
    int prepare_calls = 0;
    int predict_calls = 0;
    int unpatchify_calls = 0;
    int scheduler_calls = 0;
    int log_calls = 0;

    const bool ok = trtmc::pixart_denoising::run_pixart_video_denoising_steps(
        2, timesteps, latents, error,
        [](float timestep, std::vector<float>& temb_6d, std::vector<float>& time_embed) {
            temb_6d = {timestep};
            time_embed = {timestep + 1.0F};
        },
        [&](const std::vector<float>& current_latents, std::vector<float>& hidden) {
            ++prepare_calls;
            hidden = current_latents;
        },
        [&](const std::vector<float>& hidden, const std::vector<float>& temb_6d,
            const std::vector<float>& time_embed, std::vector<float>& output, std::string& err) {
            ++predict_calls;
            if (predict_calls == 2) {
                err = "pixart predict failed";
                return false;
            }
            output = {hidden[0] + temb_6d[0], hidden[1] + time_embed[0], hidden[2] + 1.0F};
            return true;
        },
        [&](std::vector<float>& output, std::vector<float>& noise_pred_spatial) {
            ++unpatchify_calls;
            noise_pred_spatial = output;
        },
        [&](const std::vector<float>& noise_pred_spatial, std::vector<float>& current_latents,
            int32_t) {
            ++scheduler_calls;
            for (std::size_t i = 0; i < current_latents.size(); ++i) {
                current_latents[i] += noise_pred_spatial[i];
            }
        },
        [&](int32_t, float, const std::vector<float>&) { ++log_calls; });

    check(!ok, "pixart seam propagates failure");
    check(error == "pixart predict failed", "pixart seam returns callback error");
    check(prepare_calls == 2, "pixart seam prepares hidden until failure step");
    check(predict_calls == 2, "pixart seam calls predict on failing step");
    check(unpatchify_calls == 1, "pixart seam skips unpatchify after failure");
    check(scheduler_calls == 1, "pixart seam skips scheduler after failure");
    check(log_calls == 1, "pixart seam skips logging after failure");
    check(latents == std::vector<float>({30.0F, 33.0F, 5.0F}),
          "pixart seam preserves latents from successful prior step");
}

void test_pixart_step_runner_handles_zero_steps_and_success() {
    std::vector<float> timesteps = {4.0F};
    std::vector<float> latents = {1.0F, 2.0F};
    std::string error;
    int callback_calls = 0;

    const bool zero_step_ok = trtmc::pixart_denoising::run_pixart_video_denoising_steps(
        0, timesteps, latents, error,
        [&](float, std::vector<float>&, std::vector<float>&) { ++callback_calls; },
        [&](const std::vector<float>&, std::vector<float>&) { ++callback_calls; },
        [&](const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
            std::vector<float>&, std::string&) {
            ++callback_calls;
            return true;
        },
        [&](std::vector<float>&, std::vector<float>&) { ++callback_calls; },
        [&](const std::vector<float>&, std::vector<float>&, int32_t) { ++callback_calls; },
        [&](int32_t, float, const std::vector<float>&) { ++callback_calls; });

    check(zero_step_ok, "pixart seam accepts zero steps");
    check(callback_calls == 0, "pixart seam skips callbacks for zero steps");

    int log_calls = 0;
    error.clear();
    const bool success_ok = trtmc::pixart_denoising::run_pixart_video_denoising_steps(
        1, timesteps, latents, error,
        [](float timestep, std::vector<float>& temb_6d, std::vector<float>& time_embed) {
            temb_6d = {timestep};
            time_embed = {timestep + 2.0F};
        },
        [](const std::vector<float>& current_latents, std::vector<float>& hidden) {
            hidden = current_latents;
        },
        [](const std::vector<float>& hidden, const std::vector<float>& temb_6d,
           const std::vector<float>& time_embed, std::vector<float>& output, std::string&) {
            output = {hidden[0] + temb_6d[0], hidden[1] + time_embed[0]};
            return true;
        },
        [](std::vector<float>& output, std::vector<float>& noise_pred_spatial) {
            noise_pred_spatial = output;
        },
        [](const std::vector<float>& noise_pred_spatial, std::vector<float>& current_latents,
           int32_t) {
            for (std::size_t i = 0; i < current_latents.size(); ++i) {
                current_latents[i] += noise_pred_spatial[i];
            }
        },
        [&](int32_t step, float timestep, const std::vector<float>& current_latents) {
            ++log_calls;
            check(step == 0, "pixart seam logs step index");
            check(timestep == 4.0F, "pixart seam logs timestep");
            check(current_latents == std::vector<float>({6.0F, 10.0F}),
                  "pixart seam logs updated latents");
        });

    check(success_ok, "pixart seam returns success when all callbacks succeed");
    check(error.empty(), "pixart seam leaves error empty on success");
    check(log_calls == 1, "pixart seam logs successful step");
    check(latents == std::vector<float>({6.0F, 10.0F}),
          "pixart seam applies scheduler updates on success");
}

void test_pixart_position_scale_matches_diffusers_for_runtime_grid() {
    using trtmc::diffusion::pixart_position_scale;
    check(std::abs(pixart_position_scale(32, 64, 2.0F) - 1.0F) < 1e-6F,
          "PixArt 512px grid uses unit position scale");
    check(std::abs(pixart_position_scale(64, 64, 2.0F) - 0.5F) < 1e-6F,
          "PixArt 1024px grid uses half position scale");
}

void test_pixart_dpmsolver_matches_diffusers_order_two_golden() {
    trtmc::diffusion::DPMSolverMultistepState scheduler;
    scheduler.set_timesteps(20);
    std::vector<float> sample = {0.25F};
    const std::vector<float> expected = {
        0.34548814F,  0.48291021F,  0.68116953F,  0.96267649F,  1.35352684F,
        1.88251323F,  2.57901782F,  3.46979204F,  4.57476371F,  5.87352858F,
        7.41119091F,  9.13542547F,  10.99540127F, 12.91783338F, 14.81028171F,
        16.56762662F, 18.08126019F, 19.24990672F, 19.99046704F, 20.24670902F,
    };
    for (int32_t step = 0; step < 20; ++step) {
        const float epsilon = 0.1F + 0.01F * static_cast<float>(step);
        scheduler.step(&epsilon, sample.data(), sample.data(), sample.size(), step);
        check(std::abs(sample[0] - expected[static_cast<std::size_t>(step)]) < 2e-4F,
              "PixArt DPMSolver++ order-two trace matches Diffusers");
    }
}

void test_pixart_null_attention_mask_only_keeps_eos_token() {
    const auto mask = trtmc::diffusion::make_pixart_null_attention_mask(120);
    check(mask.size() == 120, "PixArt null mask preserves text sequence length");
    check(mask[0] == 0.0F, "PixArt null mask keeps EOS token");
    check(std::all_of(mask.begin() + 1, mask.end(), [](float value) { return value == -10000.0F; }),
          "PixArt null mask rejects padding tokens");
}

void test_pixart_initial_latents_honor_override_and_requested_seed() {
    std::vector<float> latents;
    std::string error;
    const std::vector<float> supplied = {1.0F, 2.0F, 3.0F, 4.0F};
    check(trtmc::diffusion::resolve_pixart_initial_latents(supplied.size(), supplied, 99, latents,
                                                           error),
          "PixArt accepts caller-supplied initial latents");
    check(latents == supplied, "PixArt preserves caller-supplied initial latent bytes");

    error.clear();
    check(!trtmc::diffusion::resolve_pixart_initial_latents(supplied.size() + 1, supplied, 99,
                                                            latents, error),
          "PixArt rejects wrong initial latent size");
    check(error.find("initial latents") != std::string::npos,
          "PixArt reports initial latent size mismatch");

    std::vector<float> seed_one;
    std::vector<float> seed_two;
    error.clear();
    check(trtmc::diffusion::resolve_pixart_initial_latents(8, {}, 1, seed_one, error),
          "PixArt generates initial latents for seed one");
    check(trtmc::diffusion::resolve_pixart_initial_latents(8, {}, 2, seed_two, error),
          "PixArt generates initial latents for seed two");
    check(seed_one != seed_two, "PixArt requested seed changes generated initial latents");
}

} // namespace

int main() {
    test_pixart_step_runner_updates_latents_and_handles_failure();
    test_pixart_step_runner_handles_zero_steps_and_success();
    test_pixart_position_scale_matches_diffusers_for_runtime_grid();
    test_pixart_dpmsolver_matches_diffusers_order_two_golden();
    test_pixart_null_attention_mask_only_keeps_eos_token();
    test_pixart_initial_latents_honor_override_and_requested_seed();

    if (g_failures != 0) {
        std::cerr << g_failures << " PixArt denoising seam test(s) failed\n";
        return 1;
    }
    return 0;
}
