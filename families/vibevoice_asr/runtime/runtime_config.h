/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::vibevoice_asr {

struct SpecialTokenIds {
    std::int32_t im_start{-1};
    std::int32_t im_end{-1};
    std::int32_t speech_start{-1};
    std::int32_t speech_end{-1};
    std::int32_t speech_pad{-1};
    std::int32_t eos{-1};
};

// Prompt text and the ids the reference tokenizer gives it, recorded at build
// time; the native tokenizer must reproduce them before the model loads.
struct TokenizerCheck {
    std::string text;
    std::vector<std::int32_t> ids;
};

// The bundle's runtime.json, validated once at load.
struct RuntimeConfig {
    std::int32_t sample_rate{0};
    std::int32_t hop_length{0};
    std::int32_t encoder_chunk_frames{0};
    std::int32_t encoder_context_frames{0};
    double normalize_target_dbfs{0.0};
    double normalize_eps{0.0};
    std::int32_t hidden_size{0};
    std::int32_t num_layers{0};
    std::int32_t num_kv_heads{0};
    std::int32_t head_dim{0};
    std::int32_t vocab_size{0};
    std::int32_t max_sequence_length{0};
    std::int32_t max_prefill_rows{0};
    std::string system_prompt;
    std::string transcription_keys;
    SpecialTokenIds token_ids;
    std::vector<TokenizerCheck> tokenizer_checks;
};

RuntimeConfig parse_runtime_config(const std::string& json_text);

} // namespace trtmc::vibevoice_asr
