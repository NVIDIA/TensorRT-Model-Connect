/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/canary/runtime/canary_config.h"
#include "trtmc/task.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc {

constexpr int32_t CanaryMaxBeamSize = 32;

inline int32_t canary_language_token_id(const CanaryConfig& model, const std::string& language) {
    const auto it =
        std::find(model.supported_languages.begin(), model.supported_languages.end(), language);
    if (it != model.supported_languages.end()) {
        const auto index = static_cast<std::size_t>(it - model.supported_languages.begin());
        if (index < model.language_token_ids.size())
            return model.language_token_ids[index];
    }
    return -1;
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

inline void validate_canary_beam_options(const TranscriptionConfig& request) {
    if (request.beam_size < 1 || request.beam_size > CanaryMaxBeamSize) {
        throw std::invalid_argument("Canary beam_size must be in [1, " +
                                    std::to_string(CanaryMaxBeamSize) + "]");
    }
    if (!std::isfinite(request.length_penalty) || request.length_penalty < 0.0F) {
        throw std::invalid_argument("Canary length_penalty must be a finite value >= 0");
    }
}

inline void validate_canary_request_options(const TranscriptionConfig& request) {
    validate_canary_beam_options(request);
    if (request.input_sample_rate < 0) {
        throw std::invalid_argument("Canary input_sample_rate must be 0 or a positive Hz value");
    }
    if (request.task != TranscriptionTask::kTranscribe &&
        request.task != TranscriptionTask::kTranslate) {
        throw std::invalid_argument("Canary task must be kTranscribe or kTranslate");
    }
}

inline void validate_canary_language(const CanaryConfig& model, const std::string& language,
                                     std::string_view role) {
    if (language.empty() || canary_language_token_id(model, language) < 0) {
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

inline void validate_canary_nonnegative_duration(float value, const char* option_name) {
    if (!std::isfinite(value) || value < 0.0F) {
        throw std::invalid_argument(std::string("Canary ") + option_name +
                                    " must be a finite value >= 0");
    }
}

inline void validate_canary_dynamic_segment_bounds(const TranscriptionConfig& request) {
    if (request.segment_min_duration_seconds > 0.0F && request.segment_duration_seconds <= 0.0F) {
        throw std::invalid_argument(
            "Canary dynamic segmentation requires segment_duration_seconds > 0");
    }
    if (request.segment_min_duration_seconds > request.segment_duration_seconds &&
        request.segment_duration_seconds > 0.0F) {
        throw std::invalid_argument(
            "Canary segment_min_duration_seconds must not exceed segment_duration_seconds");
    }
}

inline void validate_canary_segment_overlap(const TranscriptionConfig& request) {
    const float overlap_limit = request.segment_min_duration_seconds > 0.0F
                                    ? request.segment_min_duration_seconds
                                    : request.segment_duration_seconds;
    if (request.segment_overlap_seconds > 0.0F &&
        (overlap_limit <= 0.0F || request.segment_overlap_seconds >= overlap_limit)) {
        throw std::invalid_argument(
            "Canary segment_overlap_seconds must be less than the active segment duration");
    }
    if (request.lcs_merge && request.segment_overlap_seconds <= 0.0F) {
        throw std::invalid_argument("Canary lcs_merge requires segment_overlap_seconds > 0");
    }
}

inline void validate_canary_durations(const TranscriptionConfig& request) {
    validate_canary_nonnegative_duration(request.max_input_duration_seconds,
                                         "max_input_duration_seconds");
    validate_canary_nonnegative_duration(request.segment_duration_seconds,
                                         "segment_duration_seconds");
    validate_canary_nonnegative_duration(request.segment_min_duration_seconds,
                                         "segment_min_duration_seconds");
    validate_canary_nonnegative_duration(request.segment_overlap_seconds,
                                         "segment_overlap_seconds");
    validate_canary_dynamic_segment_bounds(request);
    validate_canary_segment_overlap(request);
}

inline void validate_canary_request(const CanaryConfig& model, const TranscriptionConfig& request) {
    validate_canary_output_limits(model, request);
    validate_canary_request_options(request);
    validate_canary_language(model, request.source_language, "source");
    validate_canary_language(model, request.target_language, "target");
    validate_canary_language_pair(model, request);
    validate_canary_durations(request);
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
canary_output_control_token_ids(const CanaryConfig& model, const TranscriptionConfig& request) {
    const int32_t punctuation_id =
        request.punctuation ? model.punctuation_token_id : model.no_punctuation_token_id;
    const int32_t timestamp_id =
        request.timestamps ? model.timestamp_token_id : model.no_timestamp_token_id;
    if (punctuation_id < 0 || timestamp_id < 0) {
        throw std::invalid_argument(
            "Canary bundle tokenizer is missing punctuation or timestamp control tokens");
    }
    return {punctuation_id, timestamp_id};
}

inline std::vector<int32_t> make_canary_request_tokens(const CanaryConfig& model,
                                                       const TranscriptionConfig& request) {
    validate_canary_request(model, request);
    std::vector<int32_t> tokens = model.decoder_start_token_ids;
    validate_canary_prompt_positions(model, tokens.size());

    tokens[static_cast<std::size_t>(model.source_language_position)] =
        canary_language_token_id(model, request.source_language);
    tokens[static_cast<std::size_t>(model.target_language_position)] =
        canary_language_token_id(model, request.target_language);

    const auto [punctuation_id, timestamp_id] = canary_output_control_token_ids(model, request);
    tokens[static_cast<std::size_t>(model.punctuation_position)] = punctuation_id;
    tokens[static_cast<std::size_t>(model.timestamp_position)] = timestamp_id;
    return tokens;
}

} // namespace trtmc
