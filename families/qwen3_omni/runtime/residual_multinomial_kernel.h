/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc::qwen3_omni {

void launch_residual_exponential_race(const float* device_probabilities, std::size_t count,
                                      std::uint64_t seed, std::uint64_t draw,
                                      std::int32_t* device_token, cudaStream_t stream);

} // namespace trtmc::qwen3_omni
