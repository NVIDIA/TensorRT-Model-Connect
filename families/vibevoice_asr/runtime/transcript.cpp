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

// Complete top-level objects of a JSON array whose closing part may be missing. Only
// objects directly in the list count, the second and later ones each after a comma:
// recovery stops at the end of the list, at a missing comma, or at any other kind of
// element (a nested list, a number, a string), as a segment is an object.
nlohmann::json complete_objects(const std::string& text) {
    auto objects = nlohmann::json::array();
    int depth = 0;              // nesting inside the current top-level object
    bool expect_element = true; // after "[" or ",": an element; after an element: "," or "]"
    bool in_string = false;
    bool escaped = false;
    std::size_t start = 0;
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
        if (depth == 0) {
            if (is_space(c))
                continue;
            if (expect_element && c == '{') {
                depth = 1;
                start = i;
            } else if (!expect_element && c == ',') {
                expect_element = true;
            } else {
                break;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{' || c == '[') {
            ++depth;
        } else if ((c == '}' || c == ']') && --depth == 0) {
            auto value = nlohmann::json::parse(text.substr(start, i - start + 1), nullptr, false);
            if (value.is_discarded())
                break;
            objects.push_back(std::move(value));
            expect_element = false;
        }
    }
    return objects;
}

// The segment list (or single segment) in `body`, which must start as a JSON
// document; a list whose end is missing keeps its complete objects.
nlohmann::json parse_segments(const std::string& body) {
    if (body.empty() || (body.front() != '[' && body.front() != '{'))
        return nlohmann::json(nlohmann::json::value_t::discarded);
    auto segments = nlohmann::json::parse(body, nullptr, false);
    if (segments.is_discarded() && body.front() == '[')
        segments = complete_objects(body);
    return segments;
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
    // The role header the chat template puts before the reply: "assistant" followed by
    // whitespace (or nothing), not the start of a word such as "assistantship".
    constexpr std::string_view role = "assistant";
    if (text.compare(0, role.size(), role) == 0 &&
        (text.size() == role.size() || is_space(text[role.size()])))
        text = strip(std::string_view(text).substr(role.size()));
    ParsedTranscript result;
    result.text = text;
    // Output that starts as a JSON document is parsed as it is, so a fence marker
    // inside a segment's text stays content. Otherwise, or when that yields no
    // segment, a ```json fence around the list, as the official post-processing
    // accepts. Its closing fence is the one that ends the output, so backticks inside
    // a segment stay content; without one, the list may be cut short.
    auto segments = parse_segments(text);
    if (!segments.is_object() && !(segments.is_array() && !segments.empty())) {
        constexpr std::string_view fence = "```json";
        constexpr std::string_view close = "```";
        if (const auto open = text.find(fence); open != std::string::npos) {
            const auto start = open + fence.size();
            const bool closed = text.size() >= start + close.size() &&
                                text.compare(text.size() - close.size(), close.size(), close) == 0;
            const auto end = closed ? text.size() - close.size() : text.size();
            segments = parse_segments(strip(std::string_view(text).substr(start, end - start)));
        }
    }
    if (segments.is_object())
        segments = nlohmann::json::array({segments});
    if (!segments.is_array() || segments.empty())
        return result;
    // Every segment's content goes into `text`; `segments` holds the segments that have
    // both timestamps, as no timestamps are invented for the others.
    std::vector<TranscriptionSegment> timed;
    std::string joined;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        if (!segment.is_object() || !segment.contains("Content") ||
            !segment.at("Content").is_string())
            return result;
        const auto start = timestamp(segment, "Start", "Start time");
        const auto end = timestamp(segment, "End", "End time");
        if (!start.valid || !end.valid)
            return result;
        const auto content = segment.at("Content").get<std::string>();
        if (start.value && end.value) {
            TranscriptionSegment item;
            item.text = content;
            item.start_seconds = *start.value;
            item.end_seconds = *end.value;
            timed.push_back(std::move(item));
        }
        if (i > 0)
            joined += ' ';
        joined += content;
    }
    result.structured = true;
    result.text = strip(joined);
    result.segments = std::move(timed);
    return result;
}

} // namespace trtmc::vibevoice_asr
