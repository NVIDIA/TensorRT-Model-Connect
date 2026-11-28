/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/runtime/trt_module.h"

#include <cstdint>

namespace trtmc::openai_gpt {

// The engine dimensions that runtime.json declares.
struct EngineShape {
    std::int32_t vocab_size;
    std::int32_t num_layers;
    std::int32_t num_heads;
    std::int32_t head_dim;
    std::int32_t max_cache_length;
};

// Throws std::runtime_error unless `module`, one context of the bundle's engine, has exactly the
// tensors and profile shapes the runtime drives: token_id and position_id of 1 to `max_tokens`
// rows, key_positions of 1 to max_cache_length rows, float32 logits [1, vocab_size] and float32
// caches [1, num_heads, max_cache_length, head_dim] per layer. Nothing is sized from `shape`
// before its layer count matches the engine.
void validate_engine(const EngineShape& shape, const ITrtModule& module, std::int64_t max_tokens);

} // namespace trtmc::openai_gpt
