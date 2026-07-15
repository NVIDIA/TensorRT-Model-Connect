/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/canary/canary_config.h"
#include "trtmc/pipeline.h"
#include "trtmc/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc {

constexpr int32_t CanaryMaxBeamSize = 16;

inline int32_t canary_token_id(const ITokenizer* tokenizer, std::string_view token,
                               int32_t configured_id) {
    if (configured_id >= 0)
        return configured_id;
    if (tokenizer == nullptr)
        return -1;
    return tokenizer->id_for_token(token);
}

inline int32_t canary_language_token_id(const CanaryConfig& model, const ITokenizer* tokenizer,
                                        const std::string& language) {
    const auto it =
        std::find(model.supported_languages.begin(), model.supported_languages.end(), language);
    if (it != model.supported_languages.end()) {
        const auto index = static_cast<std::size_t>(it - model.supported_languages.begin());
        if (index < model.language_token_ids.size())
            return model.language_token_ids[index];
    }
    if (!model.supported_languages.empty())
        return -1;
    if (tokenizer == nullptr)
        return language == model.language ? model.language_token_id : -1;
    return tokenizer->id_for_token("<|" + language + "|>");
}

inline void validate_canary_output_limits(const CanaryConfig& model,
                                          const TranscriptionConfig& request) {
    if (request.max_output_tokens <= 0 ||
        (model.max_target_positions > 0 &&
         request.max_output_tokens > model.max_target_positions)) {
        throw std::invalid_argument("Canary max_output_tokens must be in [1, " +
                                    std::to_string(model.max_target_positions) + "]");
    }
}

inline void validate_canary_request_options(const TranscriptionConfig& request) {
    if (request.beam_size < 1 || request.beam_size > CanaryMaxBeamSize) {
        throw std::invalid_argument("Canary beam_size must be in [1, " +
                                    std::to_string(CanaryMaxBeamSize) + "]");
    }
    if (request.input_sample_rate < 0) {
        throw std::invalid_argument("Canary input_sample_rate must be 0 or a positive Hz value");
    }
    if (request.task != TranscriptionTask::kTranscribe &&
        request.task != TranscriptionTask::kTranslate) {
        throw std::invalid_argument("Canary task must be kTranscribe or kTranslate");
    }
}

inline void validate_canary_language(const CanaryConfig& model, const ITokenizer* tokenizer,
                                     const std::string& language, std::string_view role) {
    if (language.empty() || canary_language_token_id(model, tokenizer, language) < 0) {
        throw std::invalid_argument("Canary does not support " + std::string(role) + " language '" +
                                    language + "' in this bundle");
    }
}

inline void validate_canary_language_pair(const CanaryConfig& model,
                                          const TranscriptionConfig& request) {
    if (request.task == TranscriptionTask::kTranscribe &&
        request.source_language != request.target_language) {
        throw std::invalid_argument(
            "Canary transcribe task requires source_language == target_language; use the "
            "translate task for different languages");
    }
    if (request.task == TranscriptionTask::kTranslate) {
        if (request.source_language == request.target_language) {
            throw std::invalid_argument(
                "Canary translate task requires different source and target languages");
        }
        if (model.translation_requires_english && request.source_language != "en" &&
            request.target_language != "en") {
            throw std::invalid_argument(
                "This Canary bundle only supports translation between English and another "
                "supported language");
        }
    }
}

inline void validate_canary_durations(const TranscriptionConfig& request) {
    if (!std::isfinite(request.max_input_duration_seconds) ||
        request.max_input_duration_seconds < 0.0F) {
        throw std::invalid_argument(
            "Canary max_input_duration_seconds must be a finite value >= 0");
    }
    if (!std::isfinite(request.segment_duration_seconds) ||
        request.segment_duration_seconds < 0.0F) {
        throw std::invalid_argument("Canary segment_duration_seconds must be a finite value >= 0");
    }
}

inline void validate_canary_request(const CanaryConfig& model, const TranscriptionConfig& request,
                                    const ITokenizer* tokenizer) {
    validate_canary_output_limits(model, request);
    validate_canary_request_options(request);
    validate_canary_language(model, tokenizer, request.source_language, "source");
    validate_canary_language(model, tokenizer, request.target_language, "target");
    validate_canary_language_pair(model, request);
    validate_canary_durations(request);
}

inline std::vector<int32_t> make_canary_legacy_request_tokens(const CanaryConfig& model,
                                                              const TranscriptionConfig& request) {
    if (request.source_language != model.language || request.target_language != model.language ||
        !request.punctuation || request.timestamps ||
        request.task != TranscriptionTask::kTranscribe) {
        throw std::invalid_argument(
            "Canary bundle does not contain configurable decoder prompt metadata; rebuild "
            "it from the local checkpoint");
    }
    return {model.decoder_start_token_id, model.language_token_id, model.transcribe_token_id,
            model.notimestamps_token_id};
}

inline void validate_canary_prompt_positions(const CanaryConfig& model, std::size_t token_count) {
    const int32_t positions[] = {model.source_language_position, model.target_language_position,
                                 model.punctuation_position, model.timestamp_position};
    for (const int32_t position : positions) {
        if (position < 0 || static_cast<std::size_t>(position) >= token_count) {
            throw std::invalid_argument(
                "Canary bundle prompt metadata is incompatible with configurable decoding");
        }
    }
}

inline std::pair<int32_t, int32_t>
canary_output_control_token_ids(const CanaryConfig& model, const TranscriptionConfig& request,
                                const ITokenizer* tokenizer) {
    const int32_t punctuation_id = canary_token_id(
        tokenizer, request.punctuation ? "<|pnc|>" : "<|nopnc|>",
        request.punctuation ? model.punctuation_token_id : model.no_punctuation_token_id);
    const int32_t timestamp_id = canary_token_id(
        tokenizer, request.timestamps ? "<|timestamp|>" : "<|notimestamp|>",
        request.timestamps ? model.timestamp_token_id : model.no_timestamp_token_id);
    if (punctuation_id < 0 || timestamp_id < 0) {
        throw std::invalid_argument(
            "Canary bundle tokenizer is missing punctuation or timestamp control tokens");
    }
    return {punctuation_id, timestamp_id};
}

inline std::vector<int32_t> make_canary_request_tokens(const CanaryConfig& model,
                                                       const TranscriptionConfig& request,
                                                       const ITokenizer* tokenizer) {
    validate_canary_request(model, request, tokenizer);
    std::vector<int32_t> tokens = model.decoder_start_token_ids;
    if (tokens.empty()) {
        return make_canary_legacy_request_tokens(model, request);
    }

    validate_canary_prompt_positions(model, tokens.size());

    tokens[static_cast<std::size_t>(model.source_language_position)] =
        canary_language_token_id(model, tokenizer, request.source_language);
    tokens[static_cast<std::size_t>(model.target_language_position)] =
        canary_language_token_id(model, tokenizer, request.target_language);

    const auto [punctuation_id, timestamp_id] =
        canary_output_control_token_ids(model, request, tokenizer);
    tokens[static_cast<std::size_t>(model.punctuation_position)] = punctuation_id;
    tokens[static_cast<std::size_t>(model.timestamp_position)] = timestamp_id;
    return tokens;
}

} // namespace trtmc
