/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_asr/runtime/runtime_config.h"

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::qwen3_asr {
namespace {

std::string lower_ascii(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(std::string("qwen3_asr runtime.json: ") + message);
}

bool ids_in_vocabulary(const std::vector<std::int32_t>& ids, std::int32_t vocab_size) {
    return std::all_of(ids.begin(), ids.end(),
                       [vocab_size](std::int32_t id) { return id >= 0 && id < vocab_size; });
}

} // namespace

RuntimeConfig parse_runtime_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    RuntimeConfig config;
    auto& frontend = config.frontend;
    frontend.sample_rate = json.at("sample_rate").get<std::int32_t>();
    frontend.n_fft = json.at("n_fft").get<std::int32_t>();
    frontend.hop_length = json.at("hop_length").get<std::int32_t>();
    frontend.min_samples = json.at("min_samples").get<std::int64_t>();
    frontend.chunk_frames = json.at("chunk_frames").get<std::int32_t>();
    frontend.tokens_per_chunk = json.at("tokens_per_chunk").get<std::int32_t>();
    frontend.window_chunks = json.at("window_chunks").get<std::int32_t>();
    config.num_mel_bins = json.at("num_mel_bins").get<std::int32_t>();
    config.encoder_windows_per_launch = json.at("encoder_windows_per_launch").get<std::int32_t>();
    config.prefill_chunk_tokens = json.at("prefill_chunk_tokens").get<std::int32_t>();
    config.audio_embed_size = json.at("audio_embed_size").get<std::int32_t>();
    config.vocab_size = json.at("vocab_size").get<std::int32_t>();
    config.num_layers = json.at("num_layers").get<std::int32_t>();
    config.num_kv_heads = json.at("num_kv_heads").get<std::int32_t>();
    config.head_dim = json.at("head_dim").get<std::int32_t>();
    config.max_cache_length = json.at("max_cache_length").get<std::int32_t>();
    config.key_row_bucket = json.at("key_row_bucket").get<std::int32_t>();
    const auto cache_dtype = json.at("cache_dtype").get<std::string>();
    require(cache_dtype == "bf16" || cache_dtype == "fp32", "cache_dtype must be bf16 or fp32");
    config.cache_dtype = cache_dtype == "bf16" ? DType::kBFloat16 : DType::kFloat32;
    config.audio_token_id = json.at("audio_token_id").get<std::int32_t>();
    config.prompt_prefix_ids = json.at("prompt_prefix_ids").get<std::vector<std::int32_t>>();
    config.prompt_suffix_ids = json.at("prompt_suffix_ids").get<std::vector<std::int32_t>>();
    config.stop_token_ids = json.at("stop_token_ids").get<std::vector<std::int32_t>>();
    for (const auto& entry : json.at("languages")) {
        config.languages.push_back({entry.at("code").get<std::string>(),
                                    entry.at("name").get<std::string>(),
                                    entry.at("ids").get<std::vector<std::int32_t>>()});
    }
    config.asr_text_marker = json.at("asr_text_marker").get<std::string>();
    config.repetition_threshold = json.at("repetition_threshold").get<std::int32_t>();
    config.default_max_output_tokens = json.at("default_max_output_tokens").get<std::int64_t>();

    require(frontend.sample_rate > 0 && frontend.n_fft > 0 && frontend.hop_length > 0 &&
                frontend.min_samples >= 0,
            "invalid audio front end");
    require(frontend.chunk_frames > 0 && frontend.window_chunks > 0 &&
                frontend.tokens_per_chunk == post_cnn_length(frontend.chunk_frames),
            "invalid chunk geometry");
    require(config.num_mel_bins > 0 && config.encoder_windows_per_launch > 0 &&
                config.audio_embed_size > 0,
            "invalid encoder geometry");
    require(config.vocab_size > 0 && config.num_layers > 0 && config.num_kv_heads > 0 &&
                config.head_dim > 0 && config.max_cache_length > 0 && config.key_row_bucket > 0 &&
                config.prefill_chunk_tokens > 0 &&
                config.prefill_chunk_tokens <= config.max_cache_length,
            "invalid decoder geometry");
    require(config.audio_token_id >= 0 && config.audio_token_id < config.vocab_size,
            "audio_token_id outside the vocabulary");
    require(ids_in_vocabulary(config.prompt_prefix_ids, config.vocab_size) &&
                ids_in_vocabulary(config.prompt_suffix_ids, config.vocab_size),
            "prompt ids outside the vocabulary");
    require(!config.stop_token_ids.empty() &&
                ids_in_vocabulary(config.stop_token_ids, config.vocab_size),
            "stop ids missing or outside the vocabulary");
    for (const auto& language : config.languages)
        require(!language.code.empty() && !language.name.empty() && !language.ids.empty() &&
                    ids_in_vocabulary(language.ids, config.vocab_size),
                "invalid forced-language entry");
    require(!config.asr_text_marker.empty() && config.repetition_threshold > 0 &&
                config.default_max_output_tokens > 0,
            "invalid output parsing or generation defaults");
    return config;
}

const LanguagePrefill& find_language(const RuntimeConfig& config, std::string_view requested) {
    const auto wanted = lower_ascii(requested);
    for (const auto& language : config.languages) {
        if (wanted == lower_ascii(language.code) || wanted == lower_ascii(language.name))
            return language;
    }
    std::string names;
    for (const auto& language : config.languages)
        names += (names.empty() ? "" : ", ") + language.code + " (" + language.name + ")";
    throw std::invalid_argument("qwen3_asr cannot force source language '" +
                                std::string(requested) + "'; supported: " + names);
}

std::vector<std::int32_t> prompt_ids(const RuntimeConfig& config, std::int32_t audio_tokens,
                                     const LanguagePrefill* language) {
    if (audio_tokens <= 0)
        throw std::invalid_argument("qwen3_asr: the prompt needs at least one audio token");
    std::vector<std::int32_t> ids = config.prompt_prefix_ids;
    ids.insert(ids.end(), static_cast<std::size_t>(audio_tokens), config.audio_token_id);
    ids.insert(ids.end(), config.prompt_suffix_ids.begin(), config.prompt_suffix_ids.end());
    if (language != nullptr)
        ids.insert(ids.end(), language->ids.begin(), language->ids.end());
    return ids;
}

std::int32_t key_rows(const RuntimeConfig& config, std::int32_t valid_rows) {
    if (valid_rows <= 0 || valid_rows > config.max_cache_length)
        throw std::invalid_argument("qwen3_asr: key rows must be within the cache capacity");
    const auto bucket = static_cast<std::int64_t>(config.key_row_bucket);
    const auto rounded = (static_cast<std::int64_t>(valid_rows) + bucket - 1) / bucket * bucket;
    return static_cast<std::int32_t>(
        std::min<std::int64_t>(rounded, static_cast<std::int64_t>(config.max_cache_length)));
}

} // namespace trtmc::qwen3_asr
