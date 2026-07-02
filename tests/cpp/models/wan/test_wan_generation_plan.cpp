/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-DIFF-WAN-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-DIFF-WAN-01
// Intent:         Wan generation plan layout and scheduler mode
// Preconditions:  Wan config with valid latent dimensions
// Postconditions: Layout dimensions and scheduler parameters match expected values
// =============================================================================

#include "runtime/models/wan/wan_generation_plan.h"

#include <cmath>
#include <cstddef>
#include <iostream>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

void test_wan_generation_plan_derives_layout_and_scheduler_mode() {
    trtmc::WanDiffusionConfig config;
    config.scheduler = "ddim";
    config.num_inference_steps = 30;
    config.guidance_scale = 6.0F;
    config.video_num_frames = 81;
    config.video_height = 480;
    config.video_width = 832;
    config.scale_factor_temporal = 4;
    config.scale_factor_spatial = 8;
    config.z_dim = 16;
    config.dit_dim = 1536;
    config.text_seq_len = 512;
    config.patch_size = {1, 2, 2};
    config.flow_shift = 1.15F;

    const auto plan = trtmc::diffusion::make_wan_generation_plan(config, -1, -1.0F);

    check(plan.num_inference_steps == 30, "wan plan uses fallback steps for negative request");
    check_close(plan.guidance_scale, 6.0F, 1e-6F, "wan plan uses fallback guidance");
    check(plan.use_ddim, "wan plan selects ddim scheduler family");
    check(plan.layout.t_lat == 21, "wan plan derives temporal latent size");
    check(plan.layout.h_lat == 60 && plan.layout.w_lat == 104,
          "wan plan derives spatial latent size");
    check(plan.layout.num_patches == 21 * 30 * 52, "wan plan derives patch count");
    check(plan.layout.patch_dim == 64, "wan plan derives patch dim");
    check(plan.latent_count == static_cast<std::size_t>(16 * 21 * 60 * 104),
          "wan plan computes latent count");
}

void test_wan_flow_match_scheduler_builds_when_not_using_ddim() {
    trtmc::WanDiffusionConfig config;
    config.scheduler = "flow_match_euler";
    config.num_inference_steps = 12;
    config.flow_shift = 1.35F;

    const auto plan = trtmc::diffusion::make_wan_generation_plan(config, 8, 3.0F);
    check(!plan.use_ddim, "wan flow-match plan keeps native scheduler");
    check(plan.num_inference_steps == 8, "wan flow-match plan uses explicit request");
    check_close(plan.guidance_scale, 3.0F, 1e-6F, "wan flow-match plan uses explicit guidance");

    const auto scheduler = trtmc::diffusion::make_wan_flow_match_scheduler(plan);
    check(scheduler.timesteps.size() == 8, "wan flow-match scheduler size matches request");
    check_close(scheduler.shift, 1.35F, 1e-6F, "wan flow-match scheduler forwards shift");
}

} // namespace

int main() {
    test_wan_generation_plan_derives_layout_and_scheduler_mode();
    test_wan_flow_match_scheduler_builds_when_not_using_ddim();

    if (g_failures != 0) {
        std::cerr << g_failures << " wan generation plan test(s) failed\n";
        return 1;
    }
    return 0;
}
