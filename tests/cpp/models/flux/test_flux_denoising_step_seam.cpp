/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-FLUX-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-FLUX-01
// Intent:         Flux denoising step seam invokes callbacks and propagates failures
// Preconditions:  Flux step runner with mock forward function
// Postconditions: Latents updated after each step, failures propagated, zero steps handled
// =============================================================================

#include "runtime/models/flux/flux_denoising_step_seam.h"

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

void test_flux_step_runner_updates_latents_and_order() {
    std::vector<float> timesteps = {1000.0F, 500.0F};
    std::vector<float> latents = {1.0F, 2.0F};
    std::vector<float> hidden;
    std::vector<float> denoiser_output;
    std::string error;
    std::vector<int32_t> call_order;
    std::vector<float> logged_latents;

    const bool ok = trtmc::diffusion::run_flux_denoising_steps(
        2, timesteps, latents, hidden, denoiser_output, error,
        [&](float timestep, std::vector<float>& temb) {
            call_order.push_back(1);
            temb = {timestep};
        },
        [&](const std::vector<float>& current_latents, std::vector<float>& hidden_out) {
            call_order.push_back(2);
            hidden_out = {current_latents[0] + current_latents[1]};
        },
        [&](const std::vector<float>& hidden_in, const std::vector<float>& temb_in,
            std::vector<float>& output, std::string&) {
            call_order.push_back(3);
            output = {hidden_in[0] + temb_in[0], hidden_in[0] - temb_in[0]};
            return true;
        },
        [&](const std::vector<float>& output, std::vector<float>& velocity) {
            call_order.push_back(4);
            velocity = output;
        },
        [&](std::vector<float>& current_latents, const std::vector<float>& velocity, int32_t) {
            call_order.push_back(5);
            for (std::size_t i = 0; i < current_latents.size(); ++i) {
                current_latents[i] += velocity[i];
            }
        },
        [&](int32_t, const std::vector<float>& current_latents, const std::vector<float>&,
            const std::vector<float>&) {
            call_order.push_back(6);
            logged_latents = current_latents;
        });

    check(ok, "flux seam returns success");
    check(error.empty(), "flux seam leaves error empty");
    check(call_order.size() == 12, "flux seam invokes six callbacks per step");
    check(latents.size() == 2 && latents[0] == 1513.0F && latents[1] == -1486.0F,
          "flux seam applies scheduler updates in order");
    check(logged_latents == latents, "flux seam logs latest latents");
}

void test_flux_step_runner_handles_zero_steps_and_failure() {
    std::vector<float> timesteps = {1000.0F};
    std::vector<float> latents = {2.0F, 3.0F};
    std::vector<float> hidden;
    std::vector<float> denoiser_output;
    std::string error;
    int callback_calls = 0;

    const bool zero_step_ok = trtmc::diffusion::run_flux_denoising_steps(
        0, timesteps, latents, hidden, denoiser_output, error,
        [&](float, std::vector<float>&) { ++callback_calls; },
        [&](const std::vector<float>&, std::vector<float>&) { ++callback_calls; },
        [&](const std::vector<float>&, const std::vector<float>&, std::vector<float>&,
            std::string&) {
            ++callback_calls;
            return true;
        },
        [&](const std::vector<float>&, std::vector<float>&) { ++callback_calls; },
        [&](std::vector<float>&, const std::vector<float>&, int32_t) { ++callback_calls; },
        [&](int32_t, const std::vector<float>&, const std::vector<float>&,
            const std::vector<float>&) { ++callback_calls; });

    check(zero_step_ok, "flux seam accepts zero steps");
    check(callback_calls == 0, "flux seam skips callbacks for zero steps");

    int run_calls = 0;
    error.clear();
    const bool failure_ok = trtmc::diffusion::run_flux_denoising_steps(
        1, timesteps, latents, hidden, denoiser_output, error,
        [](float, std::vector<float>& temb) { temb = {1.0F}; },
        [](const std::vector<float>& current_latents, std::vector<float>& hidden_out) {
            hidden_out = current_latents;
        },
        [&](const std::vector<float>&, const std::vector<float>&, std::vector<float>&,
            std::string& err) {
            ++run_calls;
            err = "flux failed";
            return false;
        },
        [](const std::vector<float>&, std::vector<float>&) {},
        [](std::vector<float>&, const std::vector<float>&, int32_t) {},
        [](int32_t, const std::vector<float>&, const std::vector<float>&,
           const std::vector<float>&) {});

    check(!failure_ok, "flux seam propagates denoiser failure");
    check(run_calls == 1, "flux seam attempts failing denoiser once");
    check(error == "flux failed", "flux seam preserves denoiser error");
}

} // namespace

int main() {
    test_flux_step_runner_updates_latents_and_order();
    test_flux_step_runner_handles_zero_steps_and_failure();
    if (g_failures != 0) {
        return 1;
    }
    std::cout << "Flux denoising step seam tests passed\n";
    return 0;
}
