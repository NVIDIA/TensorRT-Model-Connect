/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vibevoice_asr/runtime/runtime_config.h"

#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::vibevoice_asr {
namespace {

std::int32_t positive_int(const nlohmann::json& raw, const char* name) {
    const auto& value = raw.at(name);
    if (!value.is_number_integer() || value.get<std::int64_t>() <= 0 ||
        value.get<std::int64_t>() > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument(std::string("VibeVoice-ASR runtime.json needs a positive "
                                                "int32 ") +
                                    name);
    return value.get<std::int32_t>();
}

std::int32_t token_id(const nlohmann::json& ids, const char* name, std::int32_t vocab) {
    const auto id = positive_int(ids, name);
    if (id >= vocab)
        throw std::invalid_argument(std::string("VibeVoice-ASR token id out of vocabulary: ") +
                                    name);
    return id;
}

} // namespace

RuntimeConfig parse_runtime_config(const std::string& json_text) {
    const auto raw = nlohmann::json::parse(json_text);
    if (raw.at("tensor_parallel_size") != 1)
        throw std::invalid_argument("VibeVoice-ASR runs on one GPU");
    RuntimeConfig config;
    config.sample_rate = positive_int(raw, "sample_rate");
    config.hop_length = positive_int(raw, "hop_length");
    config.encoder_chunk_frames = positive_int(raw, "encoder_chunk_frames");
    config.encoder_context_frames = positive_int(raw, "encoder_context_frames");
    config.normalize_target_dbfs = raw.at("normalize_target_dbfs").get<double>();
    config.normalize_eps = raw.at("normalize_eps").get<double>();
    config.hidden_size = positive_int(raw, "hidden_size");
    config.num_layers = positive_int(raw, "num_layers");
    config.num_kv_heads = positive_int(raw, "num_kv_heads");
    config.head_dim = positive_int(raw, "head_dim");
    config.vocab_size = positive_int(raw, "vocab_size");
    config.max_sequence_length = positive_int(raw, "max_sequence_length");
    config.max_prefill_rows = positive_int(raw, "max_prefill_rows");
    config.system_prompt = raw.at("system_prompt").get<std::string>();
    config.transcription_keys = raw.at("transcription_keys").get<std::string>();
    const auto& ids = raw.at("token_ids");
    config.token_ids.im_start = token_id(ids, "im_start", config.vocab_size);
    config.token_ids.im_end = token_id(ids, "im_end", config.vocab_size);
    config.token_ids.speech_start = token_id(ids, "speech_start", config.vocab_size);
    config.token_ids.speech_end = token_id(ids, "speech_end", config.vocab_size);
    config.token_ids.speech_pad = token_id(ids, "speech_pad", config.vocab_size);
    config.token_ids.eos = token_id(ids, "eos", config.vocab_size);
    for (const auto& check : raw.at("tokenizer_checks")) {
        TokenizerCheck item;
        item.text = check.at("text").get<std::string>();
        item.ids = check.at("ids").get<std::vector<std::int32_t>>();
        if (item.text.empty() || item.ids.empty())
            throw std::invalid_argument("VibeVoice-ASR tokenizer check is empty");
        config.tokenizer_checks.push_back(std::move(item));
    }
    if (config.tokenizer_checks.empty() || !(config.normalize_eps > 0.0) ||
        config.normalize_target_dbfs >= 0.0 || config.system_prompt.empty() ||
        config.transcription_keys.empty() || config.max_prefill_rows > config.max_sequence_length)
        throw std::invalid_argument("VibeVoice-ASR runtime.json is inconsistent");
    return config;
}

} // namespace trtmc::vibevoice_asr
