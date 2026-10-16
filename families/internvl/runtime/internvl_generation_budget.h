/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Host-side generation budget for the dense InternVL KV cache. Kept free of
// CUDA and TensorRT so it can be tested on its own.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace internvl {

// New tokens a request may generate before the KV cache is full. A full cache
// would otherwise slide and evict the oldest rows, including image tokens, so
// generation stops at the cache instead. A prompt the cache cannot hold is
// rejected; the message names the KV cache so callers report a capacity error.
inline int32_t generation_budget(int32_t prompt_tokens, int32_t requested_new_tokens,
                                 int32_t cache_length) {
    if (cache_length <= 0)
        return requested_new_tokens;
    if (prompt_tokens >= cache_length) {
        throw std::invalid_argument(
            "InternVL prompt needs " + std::to_string(prompt_tokens) +
            " tokens, which exceeds the bundle's KV cache length of " +
            std::to_string(cache_length) +
            "; rebuild with a larger max_sequence_length or use fewer image tiles");
    }
    return std::min(requested_new_tokens, cache_length - prompt_tokens);
}

} // namespace internvl
} // namespace trtmc
