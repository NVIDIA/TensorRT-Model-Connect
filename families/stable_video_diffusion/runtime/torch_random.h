/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <random>

namespace trtmc::stable_video_diffusion {

// The noise stream of torch.Generator("cpu").manual_seed(seed): a 32-bit
// Mersenne Twister. normal() follows ATen's portable CPU normal_fill for a
// contiguous float32 tensor of at least 16 elements: 24-bit uniforms written
// first, then Box-Muller in float32 over blocks of 16 (pairs j and j + 8), the
// last 16 values redrawn when the size is not a multiple of 16. Successive
// calls continue the stream, as successive torch.randn calls on one generator do.
class TorchCpuGenerator {
  public:
    explicit TorchCpuGenerator(std::uint32_t seed) : engine_(seed) {}

    void normal(float* values, std::size_t count);

  private:
    float uniform();

    std::mt19937 engine_;
};

} // namespace trtmc::stable_video_diffusion
