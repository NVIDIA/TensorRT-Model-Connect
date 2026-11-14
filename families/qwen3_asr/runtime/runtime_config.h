/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// runtime.json of a Qwen3-ASR bundle and the transcription prompt it describes.

#include "families/qwen3_asr/runtime/frontend.h"
#include "trtmc/runtime/tensor.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::qwen3_asr {

// Assistant prefill "language <Name><asr_text>" that forces one language.
struct LanguagePrefill {
    std::string code;
    std::string name;
    std::vector<std::int32_t> ids;
};

struct RuntimeConfig {
    FrontendConfig frontend;
    std::int32_t num_mel_bins{0};
    std::int32_t encoder_windows_per_launch{0};
    std::int32_t prefill_chunk_tokens{0};
    std::int32_t audio_embed_size{0};
    std::int32_t vocab_size{0};
    std::int32_t num_layers{0};
    std::int32_t num_kv_heads{0};
    std::int32_t head_dim{0};
    std::int32_t max_cache_length{0};
    std::int32_t key_row_bucket{0};
    DType cache_dtype{DType::kBFloat16};
    std::int32_t audio_token_id{0};
    std::vector<std::int32_t> prompt_prefix_ids;
    std::vector<std::int32_t> prompt_suffix_ids;
    std::vector<std::int32_t> stop_token_ids;
    std::vector<LanguagePrefill> languages;
    std::string asr_text_marker;
    std::int32_t repetition_threshold{0};
    std::int64_t default_max_output_tokens{0};
};

RuntimeConfig parse_runtime_config(const std::vector<char>& data);

// Resolves a language code or full name, case-insensitively, as the processor does; throws
// std::invalid_argument for a language the checkpoint was not trained to force.
const LanguagePrefill& find_language(const RuntimeConfig& config, std::string_view requested);

// Prefix ids, `audio_tokens` placeholders, suffix ids, then the forced-language prefill if any.
std::vector<std::int32_t> prompt_ids(const RuntimeConfig& config, std::int32_t audio_tokens,
                                     const LanguagePrefill* language);

// Rows of the decoder's key range for `valid_rows` valid cache rows: rounded up to the bucket,
// capped at the cache capacity.
std::int32_t key_rows(const RuntimeConfig& config, std::int32_t valid_rows);

} // namespace trtmc::qwen3_asr
