/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::chroma {

// Latent geometry of one bundle: the VAE latent [channels, height, width] and
// its 2x2-patch packing [tokens, 4 * channels].
struct LatentShape {
    std::int32_t channels{0};
    std::int32_t height{0};
    std::int32_t width{0};

    std::int32_t tokens() const { return (height / 2) * (width / 2); }
    std::size_t count() const {
        return static_cast<std::size_t>(channels) * static_cast<std::size_t>(height) *
               static_cast<std::size_t>(width);
    }
};

// Round a float to the nearest bfloat16 value (ties to even), returned as float.
float round_to_bfloat16(float value);

// FlowMatchEulerDiscreteScheduler with a static shift, as ChromaPipeline sets it:
// sigmas = linspace(1, 1/steps, steps) in float64, cast to float32, shifted in
// float32, then a terminal 0. Returns steps + 1 values.
std::vector<float> flow_match_sigmas(std::int32_t steps, float shift);

// The timestep the denoiser sees for scheduler timestep t (= sigma * 1000):
// ChromaPipeline passes (t in the model type) / 1000 and the model multiplies
// by 1000 again, each rounded to the model type (BF16 or FP32).
float conditioned_timestep(float timestep, bool bfloat16);

// get_timestep_embedding(t, 16, flip_sin_to_cos=True, downscale_freq_shift=0).
std::array<float, 16> timestep_features(float timestep);

// Chroma keeps every prompt token, </s> and the first padding token.
std::vector<float> denoiser_text_mask(const std::vector<std::int32_t>& tokenizer_mask);

// [channels, height, width] <-> [tokens, 4 * channels] (diffusers _pack_latents).
std::vector<float> pack_latents(const std::vector<float>& latents, const LatentShape& shape);
std::vector<float> unpack_latents(const std::vector<float>& packed, const LatentShape& shape);

// Gaussian noise in the latent layout, packed: the family's own initialization.
std::vector<float> initial_noise(std::uint32_t seed, const LatentShape& shape);

// The reference pipeline keeps the sampler state in the model type. With bfloat16 set, each
// function below rounds to bfloat16 exactly where ChromaPipeline does in a BF16 run; without
// it every operation is one rounded float32 operation, as in an FP32 run.

// The model-type copy of replayed or drawn initial latents.
void to_model_type(std::vector<float>& values, bool bfloat16);

// neg + scale * (pos - neg); writes into positive.
void apply_guidance(std::vector<float>& positive, const std::vector<float>& negative, float scale,
                    bool bfloat16);

// Euler step: latents + (sigma_next - sigma) * velocity.
void euler_step(std::vector<float>& latents, const std::vector<float>& velocity, float sigma,
                float sigma_next, bool bfloat16);

// latents / scaling + shift for the VAE decoder input.
void to_vae_input(std::vector<float>& latents, float scaling, float shift, bool bfloat16);

// [3, height, width] decoder output in [-1, 1] -> row-major RGB in [0, 1].
std::vector<float> to_rgb(const std::vector<float>& image, std::int32_t height, std::int32_t width,
                          bool bfloat16);

} // namespace trtmc::chroma
