/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// LongRoPE regime selection, kept free of CUDA and TensorRT so it can be tested
// on its own. The checkpoint rotates with its short factors while the sequence
// fits original_max_position_embeddings and with its long factors beyond it,
// choosing per forward call from the sequence length. On the step that first
// exceeds the threshold, the reference discards its KV cache and recomputes the
// whole sequence with the long factors.

#include <cstddef>
#include <cstdint>

namespace trtmc {

// 1 when a sequence of this length uses the long factors (threshold 0: never).
inline std::int32_t phi4_multimodal_rope_long_factor(std::size_t sequence_length,
                                                     std::int32_t threshold) {
    return threshold > 0 && sequence_length > static_cast<std::size_t>(threshold) ? 1 : 0;
}

} // namespace trtmc
