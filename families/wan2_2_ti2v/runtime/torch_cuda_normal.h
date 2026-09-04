/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::wan2_2_ti2v {

// Generate the same contiguous FP32 normal sequence as the first
// torch.randn(..., generator=torch.Generator(device="cuda").manual_seed(seed))
// call on the active CUDA device.  This is a clean native CUDA implementation;
// it does not link or call PyTorch.
std::vector<float> torch_cuda_normal(std::size_t count, uint64_t seed);

// Match Wan2.2's CUDA sinusoidal_embedding_1d(256, timestep), including its
// FP64 pow/trigonometric evaluation and final FP32 cast.
std::vector<float> torch_cuda_timestep_features(int64_t timestep);

} // namespace trtmc::wan2_2_ti2v
