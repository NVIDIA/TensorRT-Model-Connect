/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/voxtral/runtime/frontend.h"
#include "families/voxtral/runtime/text.h"
#include "trtmc/runtime/tensor.h"

#include <cstdint>
#include <vector>

namespace trtmc::voxtral {

struct RuntimeConfig {
    FrontendConfig frontend;
    PromptContract prompt;
    DType dtype{DType::kFloat32};
    std::int64_t max_audio_chunks{0};
    std::int64_t hidden_size{0};
    std::int64_t vocab_size{0};
    std::int64_t num_hidden_layers{0};
    std::int64_t kv_width{0};
    std::int64_t capacity{0};
    std::int64_t prefill_rows{0};
    std::int32_t eos_token_id{-1};
    std::int32_t num_special_tokens{0};
};

// The bundle's runtime.json. Every integer field must be a JSON integer within its range (no
// fraction, no wrap-around), no object may repeat a key, and the fixed prompt framing must hold
// valid token ids other than the audio placeholder.
RuntimeConfig parse_runtime_config(const std::vector<char>& json);

} // namespace trtmc::voxtral
