/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ltx_video/runtime/diffusion_helpers.h"
#include "families/ltx_video/runtime/pipeline.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void test_parse_ltx_video_options() {
    const std::string json = R"JSON({
      "negative_prompt": "bad frames",
      "frame_rate": 24,
      "guidance_rescale": 0.35
    })JSON";

    const auto options = trtmc::parse_ltx_video_options(json);
    check(options.negative_prompt == "bad frames", "negative prompt parsed");
    check(options.frame_rate == 24, "frame rate parsed");
    check(std::fabs(options.guidance_rescale - 0.35F) < 1e-6F, "guidance rescale parsed");
}

void test_ltx_latent_stats_parse_full_channel_count() {
    std::ostringstream mean;
    std::ostringstream stddev;
    for (int index = 0; index < 128; ++index) {
        if (index != 0) {
            mean << ',';
            stddev << ',';
        }
        mean << static_cast<float>(index) * 0.01F;
        stddev << 1.0F + static_cast<float>(index) * 0.02F;
    }

    const std::string json = std::string(R"JSON({
      "scheduler": "flow_match_euler",
      "num_inference_steps": 50,
      "guidance_scale": 3.0,
      "video_height": 480,
      "video_width": 704,
      "video_num_frames": 161,
      "z_dim": 128,
      "scale_factor_temporal": 8,
      "scale_factor_spatial": 32,
      "dit_dim": 2048,
      "dit_num_heads": 32,
      "text_seq_len": 128,
      "text_encoder_dim": 4096,
      "patch_size": [1, 1, 1],
      "diffusion_backend_type": "ltx_video",
      "latents_mean": [)JSON") +
                             mean.str() + R"JSON(],
      "latents_std": [)JSON" +
                             stddev.str() + R"JSON(]
    })JSON";

    const auto config = trtmc::make_diffusion_config(json);
    check(config.latents_mean.size() == 128, "ltx parses all latent mean channels");
    check(config.latents_std.size() == 128, "ltx parses all latent std channels");
    check(std::fabs(config.latents_mean.back() - 1.27F) < 1e-5F, "ltx parses last latent mean");
    check(std::fabs(config.latents_std.back() - 3.54F) < 1e-5F, "ltx parses last latent std");
}

} // namespace

int main() {
    test_parse_ltx_video_options();
    test_ltx_latent_stats_parse_full_channel_count();
    return 0;
}
