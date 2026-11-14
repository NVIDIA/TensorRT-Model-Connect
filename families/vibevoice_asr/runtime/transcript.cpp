/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vibevoice_asr/runtime/transcript.h"

#include <cstddef>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>

namespace trtmc::vibevoice_asr {
namespace {

bool is_space(char value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' || value == '\v' ||
           value == '\f';
}

std::string strip(std::string_view text) {
    while (!text.empty() && is_space(text.front()))
        text.remove_prefix(1);
    while (!text.empty() && is_space(text.back()))
        text.remove_suffix(1);
    return std::string(text);
}

// Complete top-level objects of a JSON array whose closing part may be missing.
nlohmann::json complete_objects(const std::string& text) {
    auto objects = nlohmann::json::array();
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    std::size_t start = std::string::npos;
    for (std::size_t i = 1; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped)
                escaped = false;
            else if (c == '\\')
                escaped = true;
            else if (c == '"')
                in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            if (depth++ == 0)
                start = i;
        } else if (c == '}' && depth > 0 && --depth == 0) {
            auto value = nlohmann::json::parse(text.substr(start, i - start + 1), nullptr, false);
            if (value.is_discarded())
                break;
            objects.push_back(std::move(value));
        } else if (depth == 0 && c == ']') {
            break;
        }
    }
    return objects;
}

// A timestamp under its key or the alias the official post-processing accepts.
// Absent: nullopt in `value`; present but not a number: `valid` is false.
struct Timestamp {
    bool valid{true};
    std::optional<double> value;
};

Timestamp timestamp(const nlohmann::json& segment, const char* key, const char* alias) {
    for (const char* name : {key, alias}) {
        const auto it = segment.find(name);
        if (it == segment.end() || it->is_null())
            continue;
        if (!it->is_number())
            return {false, std::nullopt};
        return {true, it->get<double>()};
    }
    return {};
}

} // namespace

ParsedTranscript parse_transcript(const std::string& decoded) {
    std::string text = strip(decoded);
    constexpr std::string_view role = "assistant";
    if (text.compare(0, role.size(), role) == 0)
        text = strip(std::string_view(text).substr(role.size()));
    ParsedTranscript result;
    result.text = text;
    // A ```json fence around the list, as the official post-processing accepts.
    std::string body = text;
    constexpr std::string_view fence = "```json";
    if (const auto open = body.find(fence); open != std::string::npos) {
        const auto start = open + fence.size();
        const auto close = body.find("```", start);
        body = strip(std::string_view(body).substr(
            start, close == std::string::npos ? std::string::npos : close - start));
    }
    if (body.empty() || (body.front() != '[' && body.front() != '{'))
        return result;
    auto segments = nlohmann::json::parse(body, nullptr, false);
    if (segments.is_discarded() && body.front() == '[')
        segments = complete_objects(body);
    if (segments.is_object())
        segments = nlohmann::json::array({segments});
    if (!segments.is_array() || segments.empty())
        return result;
    std::vector<TranscriptionSegment> parsed;
    std::string joined;
    bool timed = true;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        if (!segment.is_object() || !segment.contains("Content") ||
            !segment.at("Content").is_string())
            return result;
        const auto start = timestamp(segment, "Start", "Start time");
        const auto end = timestamp(segment, "End", "End time");
        if (!start.valid || !end.valid)
            return result;
        TranscriptionSegment item;
        item.text = segment.at("Content").get<std::string>();
        if (start.value && end.value) {
            item.start_seconds = *start.value;
            item.end_seconds = *end.value;
        } else {
            timed = false; // no timestamps are invented for an untimed segment
        }
        if (i > 0)
            joined += ' ';
        joined += item.text;
        parsed.push_back(std::move(item));
    }
    result.structured = true;
    result.text = strip(joined);
    if (timed)
        result.segments = std::move(parsed);
    return result;
}

} // namespace trtmc::vibevoice_asr
