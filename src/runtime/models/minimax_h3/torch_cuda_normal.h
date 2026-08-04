/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::minimax_h3 {

// Matches torch.randn(..., generator=torch.Generator().manual_seed(seed)) for
// contiguous float32 CPU tensors on AArch64. Calls are deliberately separate so
// video consumes the generator before audio, matching HF and Sol H3.
std::vector<float> torch_cuda_normal(std::size_t count, uint64_t seed, uint64_t offset = 0);
uint64_t torch_cuda_normal_consumed_offset(std::size_t count);

} // namespace trtmc::minimax_h3
