/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Host-side request math of the flux3 text-to-image flow: prompt framing, the
// flow-map schedule, timestep projections, latent packing, the Euler update and
// the image conversion. Each float32 operation is rounded separately, as in
// the reference, so this file must be compiled without FMA contraction.

#include "trtmc/runtime/span.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::flux3 {

// Rounds to the nearest bfloat16 (ties to even) and widens back to float32.
// Infinities are kept and every NaN becomes the quiet NaN 0x7FC00000.
float round_bf16(float value) noexcept;

// Values for `num_steps` steps followed by the terminal zero.
struct Schedule {
    std::vector<float> sigmas;
    std::vector<float> timesteps;
};

// Shifted float32 sigmas of numpy.linspace(1, 1 / num_steps, num_steps) and
// the timesteps sigma * 1000.
Schedule make_schedule(std::int32_t num_steps, double shift);

// The bfloat16 timestep the denoiser embeds for a scheduler timestep:
// bf16(bf16(timestep / 1000) * 1000).
float timestep_feed(float timestep) noexcept;

// Sinusoidal projection of a fed timestep: dim / 2 cosines, then dim / 2 sines.
std::vector<float> timestep_projection(float feed, std::int32_t dim, std::int32_t max_period);

std::string frame_prompt(std::string_view prefix, std::string_view prompt, std::string_view suffix);

// Token count the text engines run with: real_tokens rounded up to a multiple
// of pad_multiple. Throws std::invalid_argument above max_tokens; prompts are
// never truncated.
std::int32_t text_window(std::size_t real_tokens, std::int32_t pad_multiple,
                         std::int32_t max_tokens);

// Right-pads ids with pad_token_id to exactly `window` tokens.
std::vector<std::int32_t> pad_tokens(std::vector<std::int32_t> ids, std::int32_t window,
                                     std::int32_t pad_token_id);

struct LatentShape {
    std::int32_t channels{0};
    std::int32_t height{0};
    std::int32_t width{0};

    std::size_t count() const noexcept {
        return static_cast<std::size_t>(channels) * static_cast<std::size_t>(height) *
               static_cast<std::size_t>(width);
    }
};

// Channel-major [channels, height, width] values to bfloat16-valued tokens
// [height * width, channels], token index row * width + column.
std::vector<float> pack_latents(Span<const float> channel_major, const LatentShape& shape);

// Seeded standard normal values from std::mt19937 and
// std::normal_distribution<float>, in channel-major order.
std::vector<float> gaussian_latents(std::uint32_t seed, std::size_t count);

// One flow-map Euler step on bfloat16-valued latents, as the reference scheduler
// evaluates it: the float32 step d = next_sigma - sigma is rounded to bfloat16
// (type promotion with the bfloat16 velocity), the product is rounded to bfloat16,
// the sum is float32 and the result is rounded to bfloat16:
// x = bf16(x + bf16(bf16(next_sigma - sigma) * v)).
void euler_update(std::vector<float>& latents, Span<const float> velocity, float sigma,
                  float next_sigma);

// Decoder output [3, height, width] to HWC pixels clamp(x / 2 + 0.5, 0, 1).
// Throws std::runtime_error on a non-finite value.
std::vector<float> image_to_pixels(Span<const float> planar, std::int32_t height,
                                   std::int32_t width);

} // namespace trtmc::flux3
