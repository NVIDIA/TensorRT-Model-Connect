/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/moss_transcribe_diarize/runtime/runtime_config.h"

#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::moss_transcribe_diarize {
namespace {

std::int32_t positive(const nlohmann::json& json, const char* key) {
    const auto value = json.at(key).get<std::int64_t>();
    if (value <= 0 || value > INT32_MAX)
        throw std::runtime_error(std::string("runtime.json ") + key + " must be a positive int32");
    return static_cast<std::int32_t>(value);
}

std::vector<std::int32_t> token_ids(const nlohmann::json& json, const char* key,
                                    std::int32_t vocab_size) {
    auto ids = json.at(key).get<std::vector<std::int32_t>>();
    for (const auto id : ids) {
        if (id < 0 || id >= vocab_size)
            throw std::runtime_error(std::string("runtime.json ") + key +
                                     " holds an id outside the vocabulary");
    }
    return ids;
}

void validate_geometry(const RuntimeConfig& config) {
    if (config.precision != "fp32" && config.precision != "bf16")
        throw std::runtime_error("runtime.json precision must be fp32 or bf16");
    if (config.n_fft % 2 != 0 || config.n_fft >= config.chunk_samples)
        throw std::runtime_error("runtime.json n_fft does not fit the chunk");
    if (config.mel_frames != config.chunk_samples / config.hop_length)
        throw std::runtime_error("runtime.json mel_frames must equal chunk_samples / hop_length");
    if ((config.chunk_samples - 1) / config.samples_per_token + 1 != config.tokens_per_chunk)
        throw std::runtime_error("runtime.json full chunks must map onto tokens_per_chunk");
    if (config.num_layers <= 0 || config.prefill_chunk_tokens > config.max_sequence_length)
        throw std::runtime_error("runtime.json decoder limits are inconsistent");
    const auto& markers = config.markers;
    if (!std::isfinite(markers.audio_tokens_per_second) || markers.audio_tokens_per_second <= 0.0 ||
        markers.every_seconds <= 0)
        throw std::runtime_error("runtime.json time markers are invalid");
    if (config.prompt_prefix_ids.empty() || config.prompt_suffix_ids.empty() ||
        config.eos_token_ids.empty())
        throw std::runtime_error("runtime.json prompt or end-of-turn ids are missing");
    const auto fixed = config.prompt_prefix_ids.size() + config.prompt_suffix_ids.size();
    if (fixed + 2 > static_cast<std::size_t>(config.max_sequence_length))
        throw std::runtime_error("runtime.json sequence capacity cannot hold the prompt");
}

} // namespace

RuntimeConfig parse_runtime_config(const std::vector<char>& json_bytes) {
    const auto json = nlohmann::json::parse(json_bytes.begin(), json_bytes.end());
    RuntimeConfig config;
    config.precision = json.at("precision").get<std::string>();
    config.sample_rate = positive(json, "sample_rate");
    config.chunk_samples = positive(json, "chunk_samples");
    config.n_fft = positive(json, "n_fft");
    config.hop_length = positive(json, "hop_length");
    config.num_mel_bins = positive(json, "num_mel_bins");
    config.mel_frames = positive(json, "mel_frames");
    config.tokens_per_chunk = positive(json, "tokens_per_chunk");
    config.samples_per_token = positive(json, "samples_per_token");
    config.hidden_size = positive(json, "hidden_size");
    config.num_layers = positive(json, "num_layers");
    config.num_kv_heads = positive(json, "num_kv_heads");
    config.head_dim = positive(json, "head_dim");
    config.vocab_size = positive(json, "vocab_size");
    config.max_sequence_length = positive(json, "max_sequence_length");
    config.prefill_chunk_tokens = positive(json, "prefill_chunk_tokens");
    config.audio_token_id = json.at("audio_token_id").get<std::int32_t>();
    if (config.audio_token_id < 0 || config.audio_token_id >= config.vocab_size)
        throw std::runtime_error("runtime.json audio_token_id lies outside the vocabulary");

    const auto& markers = json.at("time_markers");
    config.markers.enabled = markers.at("enabled").get<bool>();
    config.markers.audio_tokens_per_second = markers.at("audio_tokens_per_second").get<double>();
    config.markers.every_seconds = markers.at("every_seconds").get<std::int32_t>();
    const auto digits = token_ids(markers, "digit_token_ids", config.vocab_size);
    if (digits.size() != config.markers.digit_token_ids.size())
        throw std::runtime_error("runtime.json must name one token per decimal digit");
    std::copy(digits.begin(), digits.end(), config.markers.digit_token_ids.begin());

    config.prompt_prefix_ids = token_ids(json, "prompt_prefix_ids", config.vocab_size);
    config.prompt_suffix_ids = token_ids(json, "prompt_suffix_ids", config.vocab_size);
    config.eos_token_ids = token_ids(json, "eos_token_ids", config.vocab_size);
    config.default_max_new_tokens = positive(json, "default_max_new_tokens");
    validate_geometry(config);
    return config;
}

} // namespace trtmc::moss_transcribe_diarize
