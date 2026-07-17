/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cli/args.h"
#include "trtmc/pipeline.h"

namespace trtmc::cli {

// Keep the dedicated generate-video command on the same image-generation
// contract as `trtmc run`. In particular, these fields must not stop at the
// argument parser: the model pipeline has to receive and validate them.
inline GenerateConfig make_video_generate_config(const CliArgs& args) {
    GenerateConfig config;
    config.num_steps = args.num_steps;
    config.guidance_scale = args.cfg_scale >= 0.0F ? args.cfg_scale : args.guidance_scale;
    config.seed = args.seed;
    config.negative_prompt = args.negative_prompt;
    config.height = args.diffusion_height;
    config.width = args.diffusion_width;
    return config;
}

} // namespace trtmc::cli
