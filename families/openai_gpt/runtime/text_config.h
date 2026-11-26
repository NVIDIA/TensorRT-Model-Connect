/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/openai_gpt/runtime/sampler.h"
#include "trtmc/internal/config.h"

#include <cstdint>

namespace trtmc::openai_gpt {

// The Config fields of the text_continuation Task.
Span<const internal::ConfigField> text_config_fields();

struct TextConfig {
    std::int64_t max_new_tokens{0};
    SamplingParams sampling;
};

// The checked Config of one request for a KV cache of `capacity` rows: 1 to `capacity` new
// tokens, a finite non-negative temperature, top_k in int32, top_p and min_p in [0, 1].
// Throws internal::ConfigError.
TextConfig parse_text_config(internal::ConfigView supplied, std::int32_t capacity);

} // namespace trtmc::openai_gpt
