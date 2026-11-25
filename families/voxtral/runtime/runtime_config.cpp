/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/voxtral/runtime/runtime_config.h"

#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace trtmc::voxtral {
namespace {

constexpr std::int64_t kInt32Max = std::numeric_limits<std::int32_t>::max();
// Generous bounds that keep every size product of the runtime within 64 bits.
constexpr std::int64_t kMaxWidth = std::int64_t{1} << 20;
constexpr std::int64_t kMaxLayers = 4096;

[[noreturn]] void invalid(const std::string& what) {
    throw std::invalid_argument("voxtral runtime.json: " + what);
}

nlohmann::json parse_without_repeated_keys(const std::vector<char>& data) {
    std::vector<std::unordered_set<std::string>> keys;
    bool repeated = false;
    auto document =
        nlohmann::json::parse(data.begin(), data.end(),
                              [&](int, nlohmann::json::parse_event_t event, nlohmann::json& value) {
                                  if (event == nlohmann::json::parse_event_t::object_start)
                                      keys.emplace_back();
                                  else if (event == nlohmann::json::parse_event_t::object_end)
                                      keys.pop_back();
                                  else if (event == nlohmann::json::parse_event_t::key &&
                                           !keys.back().insert(value.get<std::string>()).second)
                                      repeated = true;
                                  return true;
                              });
    if (repeated)
        invalid("an object repeats a key");
    if (!document.is_object())
        invalid("the document must be an object");
    return document;
}

// A JSON integer in [low, high]; fractions, booleans, null and out-of-range values are refused
// before any conversion.
std::int64_t integer(const nlohmann::json& value, const std::string& name, std::int64_t low,
                     std::int64_t high) {
    if (!value.is_number_integer())
        invalid(name + " must be an integer");
    std::int64_t number = 0;
    if (value.is_number_unsigned()) {
        const auto unsigned_number = value.get<std::uint64_t>();
        if (unsigned_number > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            invalid(name + " is out of range");
        number = static_cast<std::int64_t>(unsigned_number);
    } else {
        number = value.get<std::int64_t>();
    }
    if (number < low || number > high)
        invalid(name + " must lie in [" + std::to_string(low) + ", " + std::to_string(high) +
                "], got " + std::to_string(number));
    return number;
}

std::int64_t field(const nlohmann::json& json, const char* name, std::int64_t low,
                   std::int64_t high) {
    if (!json.contains(name))
        invalid(std::string(name) + " is missing");
    return integer(json.at(name), name, low, high);
}

std::vector<std::int32_t> token_ids(const nlohmann::json& value, const std::string& name,
                                    std::int64_t vocab_size, std::int32_t placeholder) {
    if (!value.is_array() || value.empty())
        invalid(name + " must be a nonempty array of token ids");
    std::vector<std::int32_t> ids;
    ids.reserve(value.size());
    for (const auto& item : value) {
        const auto id = static_cast<std::int32_t>(integer(item, name, 0, vocab_size - 1));
        if (id == placeholder)
            invalid(name + " holds the audio placeholder token");
        ids.push_back(id);
    }
    return ids;
}

} // namespace

RuntimeConfig parse_runtime_config(const std::vector<char>& data) {
    const auto json = parse_without_repeated_keys(data);
    field(json, "version", 1, 1);
    RuntimeConfig config;
    if (!json.contains("precision") || !json.at("precision").is_string())
        invalid("precision must be a string");
    const auto precision = json.at("precision").get<std::string>();
    if (precision == "fp32")
        config.dtype = DType::kFloat32;
    else if (precision == "fp16")
        config.dtype = DType::kFloat16;
    else if (precision == "bf16")
        config.dtype = DType::kBFloat16;
    else
        invalid("unsupported precision '" + precision + "'");

    // Version 1 is the WhisperFeatureExtractor front end of the Voxtral checkpoints: 16 kHz, 30 s
    // chunks, a 400-point STFT with hop 160, 3000 frames per chunk. Only the mel count follows the
    // checkpoint.
    auto& frontend = config.frontend;
    frontend.sample_rate = static_cast<std::int32_t>(field(json, "sample_rate", 16000, 16000));
    frontend.chunk_samples = field(json, "chunk_samples", 480000, 480000);
    frontend.n_fft = static_cast<std::int32_t>(field(json, "n_fft", 400, 400));
    frontend.hop_length = static_cast<std::int32_t>(field(json, "hop_length", 160, 160));
    frontend.num_mel_bins = static_cast<std::int32_t>(field(json, "num_mel_bins", 1, kMaxWidth));
    frontend.mel_frames_per_chunk =
        static_cast<std::int32_t>(field(json, "mel_frames_per_chunk", 3000, 3000));

    config.vocab_size = field(json, "vocab_size", 1, kInt32Max);
    config.hidden_size = field(json, "hidden_size", 1, kMaxWidth);
    config.num_hidden_layers = field(json, "num_hidden_layers", 1, kMaxLayers);
    config.kv_width = field(json, "kv_width", 1, kMaxWidth);
    config.capacity = field(json, "max_cache_length", 1, kInt32Max);
    config.prefill_rows = field(json, "prefill_max_rows", 1, config.capacity);
    config.eos_token_id =
        static_cast<std::int32_t>(field(json, "eos_token_id", 0, config.vocab_size - 1));
    config.num_special_tokens =
        static_cast<std::int32_t>(field(json, "num_special_tokens", 0, config.vocab_size));

    auto& prompt = config.prompt;
    prompt.audio_token_id =
        static_cast<std::int32_t>(field(json, "audio_token_id", 0, config.vocab_size - 1));
    prompt.audio_tokens_per_chunk =
        static_cast<std::int32_t>(field(json, "audio_tokens_per_chunk", 1, config.capacity));
    config.max_audio_chunks =
        field(json, "max_audio_chunks", 1, config.capacity / prompt.audio_tokens_per_chunk);
    for (const char* name : {"prompt_prefix_ids", "prompt_suffix_ids", "language_suffix_ids"}) {
        if (!json.contains(name))
            invalid(std::string(name) + " is missing");
    }
    prompt.prefix = token_ids(json.at("prompt_prefix_ids"), "prompt_prefix_ids", config.vocab_size,
                              prompt.audio_token_id);
    prompt.suffix = token_ids(json.at("prompt_suffix_ids"), "prompt_suffix_ids", config.vocab_size,
                              prompt.audio_token_id);
    const auto& languages = json.at("language_suffix_ids");
    if (!languages.is_object())
        invalid("language_suffix_ids must be an object");
    for (const auto& [language, suffix] : languages.items()) {
        if (language.empty())
            invalid("language_suffix_ids has an empty language code");
        prompt.language_suffix.emplace(language,
                                       token_ids(suffix, "language_suffix_ids." + language,
                                                 config.vocab_size, prompt.audio_token_id));
    }
    return config;
}

} // namespace trtmc::voxtral
