/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Host-side numerics of the SD3 sampler: the flow-matching sigma schedule, the
// sinusoidal timestep projection the transformer consumes, and the prompt id
// framing of the three tokenizers. Header-only so the CPU tests can include it.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc::stable_diffusion_3 {

// FlowMatchEulerDiscreteScheduler with a static shift, step for step as
// diffusers computes it: the training end points are shifted once in float32
// when the scheduler is created; set_timesteps spaces the timesteps linearly
// in float64 between them (numpy.linspace), shifts them again in float64 and
// rounds the result to float32. Timesteps are sigma * num_train_timesteps in
// float32, and a terminal sigma of zero closes the schedule.
struct FlowMatchSchedule {
    std::vector<float> sigmas;    // steps + 1 values, the last one 0
    std::vector<float> timesteps; // steps values
};

inline float shift_sigma_f32(float sigma, float shift) {
    return (shift * sigma) / (1.0F + (shift - 1.0F) * sigma);
}

inline FlowMatchSchedule flow_match_schedule(std::int32_t steps, std::int32_t num_train_timesteps,
                                             double shift) {
    if (steps < 1)
        throw std::invalid_argument("stable_diffusion_3 needs at least one sampling step");
    if (num_train_timesteps < 2 || !(shift > 0.0))
        throw std::invalid_argument("stable_diffusion_3 scheduler constants are out of range");
    const auto shift_f = static_cast<float>(shift);
    const auto train = static_cast<float>(num_train_timesteps);
    // Training sigmas run from num_train_timesteps / N down to 1 / N.
    const double sigma_max = shift_sigma_f32(train / train, shift_f);
    const double sigma_min = shift_sigma_f32(1.0F / train, shift_f);

    const double start = sigma_max * num_train_timesteps;
    const double stop = sigma_min * num_train_timesteps;
    FlowMatchSchedule schedule;
    schedule.sigmas.reserve(static_cast<std::size_t>(steps) + 1U);
    schedule.timesteps.reserve(static_cast<std::size_t>(steps));
    const double step = steps > 1 ? (stop - start) / static_cast<double>(steps - 1) : 0.0;
    for (std::int32_t index = 0; index < steps; ++index) {
        double timestep =
            steps > 1 && index == steps - 1 ? stop : static_cast<double>(index) * step + start;
        const double sigma = timestep / num_train_timesteps;
        const double shifted = shift * sigma / (1.0 + (shift - 1.0) * sigma);
        const auto sigma_f = static_cast<float>(shifted);
        schedule.sigmas.push_back(sigma_f);
        schedule.timesteps.push_back(sigma_f * train);
    }
    schedule.sigmas.push_back(0.0F);
    return schedule;
}

// get_timestep_embedding(t, channels, flip_sin_to_cos=True, downscale_freq_shift=0):
// [cos(t * f_j), sin(t * f_j)] with f_j = exp(-ln(10000) * j / (channels / 2)).
inline void timestep_projection(float timestep, std::int32_t channels, float* out) {
    if (channels <= 0 || channels % 2 != 0)
        throw std::invalid_argument("stable_diffusion_3 timestep channels must be even");
    const std::int32_t half = channels / 2;
    const auto coefficient = static_cast<float>(-std::log(10000.0));
    for (std::int32_t j = 0; j < half; ++j) {
        const float exponent = coefficient * static_cast<float>(j) / static_cast<float>(half);
        const float argument = timestep * std::exp(exponent);
        out[j] = std::cos(argument);
        out[half + j] = std::sin(argument);
    }
}

// CLIP framing: <|startoftext|> content <|endoftext|>, then pad to length.
// The content is truncated so that both brackets fit.
inline std::vector<std::int32_t> frame_clip_ids(std::vector<std::int32_t> content, std::int32_t bos,
                                                std::int32_t eos, std::int32_t pad,
                                                std::size_t length) {
    if (length < 2)
        throw std::invalid_argument("stable_diffusion_3 CLIP length is too short");
    if (content.size() > length - 2U)
        content.resize(length - 2U);
    std::vector<std::int32_t> ids;
    ids.reserve(length);
    ids.push_back(bos);
    ids.insert(ids.end(), content.begin(), content.end());
    ids.push_back(eos);
    ids.resize(length, pad);
    return ids;
}

// The pooled CLIP position. transformers reads checkpoints whose config
// still names eos_token_id 2 at the first maximum id (the end-of-text token
// has the largest id), and newer ones at the first eos_token_id.
inline std::int32_t clip_pool_index(const std::vector<std::int32_t>& ids,
                                    std::int32_t config_eos_token_id) {
    if (ids.empty())
        throw std::invalid_argument("stable_diffusion_3 CLIP ids are empty");
    if (config_eos_token_id == 2)
        return static_cast<std::int32_t>(std::max_element(ids.begin(), ids.end()) - ids.begin());
    const auto found = std::find(ids.begin(), ids.end(), config_eos_token_id);
    // argmax over an all-false mask is 0, as in transformers.
    return found == ids.end() ? 0 : static_cast<std::int32_t>(found - ids.begin());
}

// T5 framing: content </s>, then pad to length; the content is truncated so
// that </s> fits.
inline std::vector<std::int32_t> frame_t5_ids(std::vector<std::int32_t> content, std::int32_t eos,
                                              std::int32_t pad, std::size_t length) {
    if (length < 1)
        throw std::invalid_argument("stable_diffusion_3 T5 length is too short");
    if (content.size() > length - 1U)
        content.resize(length - 1U);
    content.push_back(eos);
    content.resize(length, pad);
    return content;
}

} // namespace trtmc::stable_diffusion_3
