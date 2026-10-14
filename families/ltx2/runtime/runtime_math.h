/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Pure host-side helpers of the LTX-2.5 runtime (header-only so the contract tests compile
// them without engines).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {

// Euler step of diffusers FlowMatchEulerDiscreteScheduler: x + (sigma_next - sigma) * v.
inline void ltx2_euler_step(std::vector<float>& x, const std::vector<float>& v, float sigma,
                            float sigma_next) {
    if (x.size() != v.size())
        throw std::runtime_error("LTX-2.5 scheduler step: latent and velocity sizes differ");
    const float dt = sigma_next - sigma;
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = x[i] + dt * v[i];
}

// Two-stage re-noise (diffusers LTX2Pipeline._create_noised_state):
// x = noise_scale * noise + (1 - noise_scale) * x.
inline void ltx2_renoise(std::vector<float>& x, const std::vector<float>& noise,
                         float noise_scale) {
    if (x.size() != noise.size())
        throw std::runtime_error("LTX-2.5 re-noise: latent and noise sizes differ");
    const float keep = 1.0F - noise_scale;
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = noise_scale * noise[i] + keep * x[i];
}

// Gemma prompt ids for one prompt: the tokenizer ids (no special tokens) right-truncated to
// seq_len and left-padded with pad_id, as LTX2Pipeline tokenizes with padding_side="left".
// mask is 1 on tokens and 0 on padding.
inline void ltx2_prompt_ids(const std::vector<int32_t>& ids, int32_t seq_len, int32_t pad_id,
                            std::vector<int32_t>& padded, std::vector<int32_t>& mask) {
    const auto n = std::min<std::size_t>(ids.size(), static_cast<std::size_t>(seq_len));
    padded.assign(static_cast<std::size_t>(seq_len), pad_id);
    mask.assign(static_cast<std::size_t>(seq_len), 0);
    const auto offset = static_cast<std::size_t>(seq_len) - n;
    for (std::size_t i = 0; i < n; ++i) {
        padded[offset + i] = ids[i];
        mask[offset + i] = 1;
    }
}

// Planar [channels][samples] -> interleaved [samples][channels].
inline std::vector<float> ltx2_interleave(const std::vector<float>& planar, int32_t channels) {
    if (channels <= 0 || planar.size() % static_cast<std::size_t>(channels) != 0)
        throw std::runtime_error("LTX-2.5 audio has an invalid channel layout");
    const auto samples = planar.size() / static_cast<std::size_t>(channels);
    std::vector<float> out(planar.size());
    for (std::size_t s = 0; s < samples; ++s)
        for (int32_t c = 0; c < channels; ++c)
            out[s * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)] =
                planar[static_cast<std::size_t>(c) * samples + s];
    return out;
}

} // namespace trtmc
