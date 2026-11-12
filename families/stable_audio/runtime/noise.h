/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// The family's Gaussian noise, a documented counter-based generator so that the
// family tests can draw the very same values and hand them to the reference
// pipeline (whose own sampler draws from a torchsde Brownian tree).
//
//   key        = mix64(seed + G * (stream + 1))
//   word(k)    = mix64(key + G * (k + 1))                 G = 0x9E3779B97F4A7C15
//   u1         = ((word(2p) >> 11) + 1) * 2^-53           in (0, 1]
//   u2         = (word(2p + 1) >> 11) * 2^-53             in [0, 1)
//   value(2p)  = float(sqrt(-2 ln u1) * cos(2 pi u2))     Box-Muller in double
//   value(2p+1)= float(sqrt(-2 ln u1) * sin(2 pi u2))
//
// mix64 is the SplitMix64 finalizer. All integer arithmetic wraps modulo 2^64.
// Stream 0 is the initial latent noise, stream i + 1 the noise of step i.

#include <cstddef>
#include <cstdint>

namespace trtmc::stable_audio {

std::uint64_t mix64(std::uint64_t value) noexcept;

void fill_gaussian(std::uint64_t seed, std::uint64_t stream, float* out, std::size_t count);

} // namespace trtmc::stable_audio
