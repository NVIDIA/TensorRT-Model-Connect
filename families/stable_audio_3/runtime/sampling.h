/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Host-side sampling arithmetic of the Stable Audio 3 pipeline: render sizing,
// the timestep schedule and the family's Gaussian noise. Free of CUDA so that
// it can be unit tested on a CPU.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace trtmc::stable_audio_3 {

struct RenderPlan {
    std::int32_t latents{0};           // latent length the DiT and the decoder run at
    std::int64_t effective_latents{0}; // latents covering the duration (schedule shift)
    std::int32_t valid_latents{0};     // effective + headroom, capped at the render length
    std::int64_t output_samples{0};    // samples per channel returned to the caller
};

struct RenderLimits {
    double sample_rate{44100.0};
    std::int64_t downsampling_ratio{4096};
    std::int64_t latent_align{2};
    std::int64_t max_latents{1292};
    double headroom_seconds{6.0};
    std::int64_t headroom_latents{64};
};

// Longest accepted duration; the render itself stops at max_latents.
inline constexpr double kMaxDurationSeconds = 1.0e9;

// StableAudioModel.generate: _adapt_sample_size, compute_effective_seq_len and
// the output truncation for one positive duration.
inline RenderPlan plan_render(double seconds, const RenderLimits& limits) {
    if (!(seconds > 0.0) || !(seconds <= kMaxDurationSeconds))
        throw std::invalid_argument("stable_audio_3 duration must be in (0, 1e9] seconds");
    const auto ratio = limits.downsampling_ratio;
    auto samples =
        static_cast<std::int64_t>((seconds + limits.headroom_seconds) * limits.sample_rate);
    samples = (samples + ratio - 1) / ratio * ratio;
    const auto align = ratio * limits.latent_align;
    samples = (samples + align - 1) / align * align;
    samples = std::min(samples, limits.max_latents * ratio);
    RenderPlan plan;
    plan.latents = static_cast<std::int32_t>(samples / ratio);
    const auto audio = static_cast<std::int64_t>(seconds * limits.sample_rate);
    plan.effective_latents = static_cast<std::int64_t>(
        std::ceil(static_cast<double>(audio) / static_cast<double>(ratio)));
    plan.valid_latents = static_cast<std::int32_t>(
        std::min<std::int64_t>(plan.effective_latents + limits.headroom_latents, plan.latents));
    plan.output_samples = std::min<std::int64_t>(audio, samples);
    return plan;
}

// torch.linspace(start, end, steps) in float32, evaluated from both ends.
inline std::vector<float> linspace_f32(float start, float end, std::int32_t steps) {
    std::vector<float> values(static_cast<std::size_t>(steps));
    if (steps == 1) {
        values[0] = start;
        return values;
    }
    const float step = (end - start) / static_cast<float>(steps - 1);
    const std::int32_t halfway = steps / 2;
    for (std::int32_t index = 0; index < steps; ++index)
        values[static_cast<std::size_t>(index)] =
            index < halfway ? start + step * static_cast<float>(index)
                            : end - step * static_cast<float>(steps - index - 1);
    return values;
}

struct LogSnrShift {
    float anchor_length{2000.0F};
    float anchor_logsnr{-6.2F};
    float rate{0.0F};
    float logsnr_end{2.0F};
};

// build_schedule(steps, sigma_max=1, dist_shift=LogSNRShift, effective_seq_len)
// of the reference: steps + 1 times from 1 to 0.
inline std::vector<float> schedule(std::int32_t steps, std::int64_t effective_latents,
                                   const LogSnrShift& shift) {
    if (steps < 1)
        throw std::invalid_argument("stable_audio_3 steps must be at least 1");
    auto t = linspace_f32(1.0F, 0.0F, steps + 1);
    const float seq_len = static_cast<float>(std::max<std::int64_t>(effective_latents, 1));
    const float log2_ratio = std::log2(seq_len / shift.anchor_length);
    const float start = shift.anchor_logsnr - shift.rate * log2_ratio;
    const float span = shift.logsnr_end - start;
    for (auto& value : t) {
        if (value <= 0.0F) {
            value = 0.0F;
            continue;
        }
        if (value >= 1.0F) {
            value = 1.0F;
            continue;
        }
        const float logsnr = shift.logsnr_end - value * span;
        value = static_cast<float>(1.0 / (1.0 + std::exp(static_cast<double>(logsnr))));
    }
    t.front() = 1.0F;
    return t;
}

// Round a float to the nearest IEEE half (ties to even) and widen it again.
inline float round_to_half(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = bits & 0x80000000U;
    const std::uint32_t magnitude = bits & 0x7FFFFFFFU;
    if (magnitude >= 0x7F800000U)
        return value; // inf / nan stay
    float result = 0.0F;
    const float absolute = std::fabs(value);
    if (absolute >= 65520.0F) {
        result = INFINITY;
    } else if (absolute < 6.103515625e-05F) {
        // Half subnormals are multiples of 2^-24.
        result = std::nearbyint(absolute * 16777216.0F) / 16777216.0F;
    } else {
        // 10 explicit mantissa bits: round the 23-bit float mantissa to 10 bits.
        const std::uint32_t lsb = (magnitude >> 13) & 1U;
        const std::uint32_t rounded = (magnitude + 0x0FFFU + lsb) & ~0x1FFFU;
        std::memcpy(&result, &rounded, sizeof(result));
    }
    std::uint32_t out = 0;
    std::memcpy(&out, &result, sizeof(out));
    out |= sign;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

// The family's Gaussian noise: SplitMix64 counters and the Box-Muller transform.
// Draw d of a render uses its own stream; draw 0 is the initial latent noise and
// draw i + 1 the noise of sampler step i. The Python test harness implements the
// same function, so a reference run can be fed the identical noise.
inline std::uint64_t splitmix64(std::uint64_t value) {
    std::uint64_t z = value + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

inline void gaussian_noise(std::uint64_t seed, std::uint64_t draw, float* out, std::size_t count) {
    const std::uint64_t key = splitmix64(splitmix64(seed) + draw);
    constexpr double kScale = 1.0 / 9007199254740992.0; // 2^-53
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    for (std::size_t pair = 0; 2 * pair < count; ++pair) {
        const std::uint64_t first = splitmix64(key + 2 * pair);
        const std::uint64_t second = splitmix64(key + 2 * pair + 1);
        const double u1 = static_cast<double>((first >> 11U) + 1U) * kScale;
        const double u2 = static_cast<double>(second >> 11U) * kScale;
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = kTwoPi * u2;
        out[2 * pair] = static_cast<float>(radius * std::cos(angle));
        if (2 * pair + 1 < count)
            out[2 * pair + 1] = static_cast<float>(radius * std::sin(angle));
    }
}

} // namespace trtmc::stable_audio_3
