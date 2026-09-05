/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ltx_video/runtime/ltx_video_scheduler_helpers.h"

#include <cmath>
#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++failures;
    }
}

void test_dynamic_flow_match_scheduler_fields() {
    trtmc::diffusion::ltx_video_scheduler::FlowMatchEulerState scheduler;
    scheduler.num_train_timesteps = 1000;
    scheduler.use_dynamic_shifting = true;
    scheduler.base_shift = 0.95F;
    scheduler.max_shift = 2.05F;
    scheduler.base_image_seq_len = 1024;
    scheduler.max_image_seq_len = 4096;
    scheduler.shift_terminal = 0.1F;
    scheduler.image_seq_len = 21 * 15 * 22;

    scheduler.set_timesteps(50);

    check(scheduler.last_used_dynamic_shifting, "ltx scheduler uses dynamic shifting");
    check(scheduler.sigmas.size() == 51, "ltx scheduler appends terminal sigma");
    check_close(static_cast<float>(scheduler.sigmas[49]), 0.1F, 1e-5F,
                "ltx scheduler stretches final pre-terminal sigma");
    check_close(static_cast<float>(scheduler.sigmas[50]), 0.0F, 1e-6F,
                "ltx scheduler terminal sigma remains zero");
    check_close(static_cast<float>(scheduler.last_dynamic_mu), 3.06478F, 1e-4F,
                "ltx scheduler uses LTX base/max sequence shift formula");
}

void test_dynamic_schedule_matches_diffusers_training_sigma_bounds() {
    trtmc::diffusion::ltx_video_scheduler::FlowMatchEulerState scheduler;
    scheduler.num_train_timesteps = 1000;
    scheduler.use_dynamic_shifting = true;
    scheduler.base_shift = 0.95F;
    scheduler.max_shift = 2.05F;
    scheduler.base_image_seq_len = 1024;
    scheduler.max_image_seq_len = 4096;
    scheduler.shift_terminal = 0.1F;
    scheduler.image_seq_len = 128;

    scheduler.set_timesteps(30);

    check_close(scheduler.timesteps[1], 983.172302F, 1e-4F,
                "ltx scheduler matches Diffusers near schedule head");
    check_close(scheduler.timesteps[19], 546.948364F, 1e-4F,
                "ltx scheduler matches Diffusers schedule midpoint");
    check_close(scheduler.timesteps[29], 100.0F, 1e-5F,
                "ltx scheduler matches Diffusers terminal stretch");
}

} // namespace

int main() {
    test_dynamic_flow_match_scheduler_fields();
    test_dynamic_schedule_matches_diffusers_training_sigma_bounds();
    return failures == 0 ? 0 : 1;
}
