/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::minimax_h3 {

// Host implementation of the Philox subsequence/layout used by a fresh CUDA
// PyTorch generator for contiguous BF16 normal noise. The current CUDA device's
// launch limits determine the subsequence-to-element mapping; no GPU kernel is
// launched. Each call starts at offset zero and owns its independent seed.
// Returned FP32 values are BF16-representable, ready for FP32 engine bindings.
// Host transcendental functions can differ from CUDA's device intrinsics near
// BF16 rounding boundaries; this is not a cross-platform bitwise RNG guarantee.
std::vector<float> make_minimax_h3_turbo_noise(std::size_t count, std::uint64_t seed);

} // namespace trtmc::minimax_h3
