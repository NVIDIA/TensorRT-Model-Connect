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
#include <vector>

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
    config.video_num_frames = 5;
    config.video_height = 64;
    config.video_width = 64;
    config.scale_factor_temporal = 4;
    config.scale_factor_spatial = 8;
    config.z_dim = 16;
    config.dit_dim = 1536;
    config.text_seq_len = 226;
    config.patch_size = {1, 2, 2};

    const auto plan = trtmc::diffusion::make_wan_generation_plan(config, 8, 3.0F);
    check(!plan.use_ddim, "wan flow-match plan keeps native scheduler");
    check(plan.num_inference_steps == 8, "wan flow-match plan uses explicit request");
    check_close(plan.guidance_scale, 3.0F, 1e-6F, "wan flow-match plan uses explicit guidance");

    const auto scheduler = trtmc::diffusion::make_wan_flow_match_scheduler(plan);
    check(scheduler.timesteps.size() == 8, "wan flow-match scheduler size matches request");
    check_close(scheduler.shift, 1.35F, 1e-6F, "wan flow-match scheduler forwards shift");
}

void test_wan_unipc_flow_scheduler_matches_diffusers_order_two_reference() {
    trtmc::WanDiffusionConfig config;
    config.scheduler = "unipc_multistep";
    config.num_inference_steps = 3;
    config.flow_shift = 3.0F;
    config.video_num_frames = 5;
    config.video_height = 64;
    config.video_width = 64;
    config.scale_factor_temporal = 4;
    config.scale_factor_spatial = 8;
    config.z_dim = 16;
    config.dit_dim = 1536;
    config.text_seq_len = 226;
    config.patch_size = {1, 2, 2};

    const auto plan = trtmc::diffusion::make_wan_generation_plan(config, -1, -1.0F);
    check(plan.use_unipc, "wan plan selects checkpoint UniPC scheduler");
    auto scheduler = trtmc::diffusion::make_wan_unipc_scheduler(config, plan);

    check(scheduler.timesteps.size() == 3, "wan UniPC timestep count matches request");
    check_close(scheduler.timesteps[0], 999.0F, 1e-6F,
                "wan UniPC nudges and truncates first timestep like Diffusers");
    check_close(scheduler.timesteps[1], 857.0F, 1e-6F,
                "wan UniPC shifted middle timestep matches Diffusers");
    check_close(scheduler.timesteps[2], 600.0F, 1e-6F,
                "wan UniPC final model timestep matches Diffusers");
    check_close(scheduler.sigmas[1], 0.8573265075683594F, 1e-7F,
                "wan UniPC shifted sigma matches Diffusers");

    std::vector<float> sample = {0.25F, -0.5F};
    const std::vector<std::vector<float>> velocities = {
        {0.1F, -0.2F},
        {0.3F, 0.4F},
        {-0.1F, 0.2F},
    };
    const std::vector<std::vector<float>> expected = {
        {0.235732764005661F, -0.471465528011322F},
        {0.147221013903618F, -0.608697295188904F},
        {0.207292959094048F, -0.728841185569763F},
    };
    for (int32_t step = 0; step < 3; ++step) {
        scheduler.step(velocities[static_cast<std::size_t>(step)].data(), sample.data(),
                       sample.data(), sample.size(), step);
        check_close(sample[0], expected[static_cast<std::size_t>(step)][0], 2e-6F,
                    "wan UniPC first value matches Diffusers");
        check_close(sample[1], expected[static_cast<std::size_t>(step)][1], 2e-6F,
                    "wan UniPC second value matches Diffusers");
    }
}

} // namespace

int main() {
    test_wan_generation_plan_derives_layout_and_scheduler_mode();
    test_wan_flow_match_scheduler_builds_when_not_using_ddim();
    test_wan_unipc_flow_scheduler_matches_diffusers_order_two_reference();

    if (g_failures != 0) {
        std::cerr << g_failures << " wan generation plan test(s) failed\n";
        return 1;
    }
    return 0;
}
