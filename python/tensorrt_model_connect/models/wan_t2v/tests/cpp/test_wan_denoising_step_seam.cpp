/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-WAN-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-WAN-01
// Intent:         Wan-owned denoising step runner updates latents correctly
// Preconditions:  Wan step runner with mock forward function
// Postconditions: Latents updated after each step, failures propagated, zero steps handled
// =============================================================================

#include "wan_denoising_step_seam.h"
#include "wan_matmul_policy.h"

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

void test_wan_step_runner_updates_latents_and_handles_failure() {
    std::vector<float> timesteps = {30.0F, 20.0F};
    std::vector<float> latents = {0.0F, 1.0F, 2.0F};
    std::string error;
    int prepare_calls = 0;
    int predict_calls = 0;
    int unpatchify_calls = 0;
    int scheduler_calls = 0;
    int log_calls = 0;

    const bool ok = trtmc::wan_denoising::run_wan_video_denoising_steps(
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
                err = "wan predict failed";
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

    check(!ok, "wan seam propagates failure");
    check(error == "wan predict failed", "wan seam returns callback error");
    check(prepare_calls == 2, "wan seam prepares hidden until failure step");
    check(predict_calls == 2, "wan seam calls predict on failing step");
    check(unpatchify_calls == 1, "wan seam skips unpatchify after failure");
    check(scheduler_calls == 1, "wan seam skips scheduler after failure");
    check(log_calls == 1, "wan seam skips logging after failure");
    check(latents == std::vector<float>({30.0F, 33.0F, 5.0F}),
          "wan seam preserves latents from successful prior step");
}

void test_wan_step_runner_handles_zero_steps_and_success() {
    std::vector<float> timesteps = {4.0F};
    std::vector<float> latents = {1.0F, 2.0F};
    std::string error;
    int callback_calls = 0;

    const bool zero_step_ok = trtmc::wan_denoising::run_wan_video_denoising_steps(
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

    check(zero_step_ok, "wan seam accepts zero steps");
    check(callback_calls == 0, "wan seam skips callbacks for zero steps");

    int log_calls = 0;
    error.clear();
    const bool success_ok = trtmc::wan_denoising::run_wan_video_denoising_steps(
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
            check(step == 0, "wan seam logs step index");
            check(timestep == 4.0F, "wan seam logs timestep");
            check(current_latents == std::vector<float>({6.0F, 10.0F}),
                  "wan seam logs updated latents");
        });

    check(success_ok, "wan seam returns success when all callbacks succeed");
    check(error.empty(), "wan seam leaves error empty on success");
    check(log_calls == 1, "wan seam logs successful step");
    check(latents == std::vector<float>({6.0F, 10.0F}),
          "wan seam applies scheduler updates on success");
}

void test_wan_large_preprocessing_matmuls_use_gpu() {
    check(trtmc::wan_should_use_gpu_matmul(7800, 64, 1536),
          "wan routes the release patch projection to GPU");
    check(trtmc::wan_should_use_gpu_matmul(226, 4096, 1536), "wan routes text projection to GPU");
    check(trtmc::wan_should_use_gpu_matmul(1, 1536, 9216),
          "wan routes large timestep projection to GPU");
    check(!trtmc::wan_should_use_gpu_matmul(1, 256, 1536),
          "wan keeps small timestep projection on CPU");
    check(!trtmc::wan_should_use_gpu_matmul(0, 64, 1536), "wan rejects invalid matrix dimensions");
}

} // namespace

int main() {
    test_wan_step_runner_updates_latents_and_handles_failure();
    test_wan_step_runner_handles_zero_steps_and_success();
    test_wan_large_preprocessing_matmuls_use_gpu();

    if (g_failures != 0) {
        std::cerr << g_failures << " Wan denoising seam test(s) failed\n";
        return 1;
    }
    return 0;
}
