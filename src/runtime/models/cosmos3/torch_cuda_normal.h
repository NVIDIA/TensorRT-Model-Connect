/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::cosmos3 {

// Generate the CUDA Philox normal sequence used by the first BF16
// torch.randn(..., generator=torch.Generator(device="cuda").manual_seed(seed))
// call and return each BF16 value losslessly in an FP32 host slot.
std::vector<float> torch_cuda_normal(std::size_t count, uint64_t seed);

// Match Diffusers Timesteps(256, flip_sin_to_cos=True,
// downscale_freq_shift=0) after Cosmos3 applies timestep_scale=0.001.
std::vector<float> torch_cuda_timestep_features(int64_t timestep);

} // namespace trtmc::cosmos3
