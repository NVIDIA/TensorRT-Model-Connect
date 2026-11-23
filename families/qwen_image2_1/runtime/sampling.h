/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/qwen_image2_1/runtime/runtime_config.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::qwen_image2_1 {

// Sigmas of the pipeline's default schedule for `num_steps` steps: linspace(1, 1 / num_steps),
// exponential shift by mu(image tokens), stretch to shift_terminal, then the terminal 0.
// The result has num_steps + 1 entries. Throws std::invalid_argument for a step count the
// schedule does not define.
std::vector<float> flow_match_sigmas(const RuntimeConfig& config, std::int32_t num_steps);

// Nearest BF16 value, ties to even, returned as float.
float round_to_bfloat16(float value);

// The timestep the transformer receives for `sigma` when the reference pipeline runs in BF16:
// the scheduler's timestep sigma * num_train_timesteps and its division by 1000 are both BF16
// values. Bundles exist only for num_train_timesteps = 1000, where the two factors are the same.
float reference_timestep(float sigma, std::int32_t num_train_timesteps);

// latents += (sigma_next - sigma) * velocity
void euler_step(std::vector<float>& latents, const std::vector<float>& velocity, float sigma,
                float sigma_next);

// Standard-normal initial latents for a seed.
std::vector<float> gaussian_latents(std::size_t count, std::uint64_t seed);

// [channels, height, width] <-> [height * width, channels], row-major tokens.
std::vector<float> pack_latents(const float* chw, std::int32_t channels, std::int32_t height,
                                std::int32_t width);
std::vector<float> unpack_latents(const std::vector<float>& tokens, std::int32_t channels,
                                  std::int32_t height, std::int32_t width);

// [channels, height, width] in [-1, 1] -> [height, width, channels] in [0, 1].
std::vector<float> image_to_hwc_unit(const float* chw, std::int32_t channels, std::int32_t height,
                                     std::int32_t width);

} // namespace trtmc::qwen_image2_1
