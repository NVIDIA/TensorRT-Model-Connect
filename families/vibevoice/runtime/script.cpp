/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vibevoice/runtime/script.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <stdexcept>

namespace trtmc::vibevoice {
namespace {

bool is_space(char value) {
    return std::isspace(static_cast<unsigned char>(value)) != 0;
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && is_space(text.front()))
        text.remove_prefix(1);
    while (!text.empty() && is_space(text.back()))
        text.remove_suffix(1);
    return text;
}

// ``Speaker <digits> : <text>`` with case-insensitive "Speaker" and optional spaces.
std::optional<Turn> speaker_line(std::string_view line) {
    constexpr std::string_view kWord = "speaker";
    if (line.size() < kWord.size())
        return std::nullopt;
    for (std::size_t index = 0; index < kWord.size(); ++index) {
        if (std::tolower(static_cast<unsigned char>(line[index])) != kWord[index])
            return std::nullopt;
    }
    std::size_t cursor = kWord.size();
    if (cursor >= line.size() || !is_space(line[cursor]))
        return std::nullopt;
    while (cursor < line.size() && is_space(line[cursor]))
        ++cursor;
    const std::size_t digits = cursor;
    std::int64_t speaker = 0;
    while (cursor < line.size() && std::isdigit(static_cast<unsigned char>(line[cursor])) != 0) {
        speaker = speaker * 10 + (line[cursor] - '0');
        if (speaker > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument("vibevoice speaker id is too large");
        ++cursor;
    }
    if (cursor == digits)
        return std::nullopt;
    while (cursor < line.size() && is_space(line[cursor]))
        ++cursor;
    if (cursor >= line.size() || line[cursor] != ':')
        return std::nullopt;
    return Turn{static_cast<std::int32_t>(speaker), std::string(trim(line.substr(cursor + 1)))};
}

} // namespace

std::vector<Turn> parse_script(std::string_view text) {
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = std::min(text.find('\n', start), text.size());
        const auto line = trim(text.substr(start, end - start));
        if (!line.empty())
            lines.push_back(line);
        start = end + 1;
    }
    if (lines.empty())
        throw std::invalid_argument("vibevoice requires non-empty text");
    std::vector<Turn> turns;
    for (const auto line : lines) {
        if (auto turn = speaker_line(line))
            turns.push_back(std::move(*turn));
    }
    if (turns.empty())
        return {Turn{0, std::string(trim(text))}};
    if (turns.size() != lines.size())
        throw std::invalid_argument(
            "vibevoice scripts need every non-empty line to read 'Speaker <id>: <text>'");
    for (const auto& turn : turns) {
        if (turn.text.empty())
            throw std::invalid_argument("vibevoice script has a speaker line without text");
    }
    const auto lowest =
        std::min_element(turns.begin(), turns.end(), [](const Turn& a, const Turn& b) {
            return a.speaker < b.speaker;
        })->speaker;
    if (lowest > 0) {
        for (auto& turn : turns)
            turn.speaker -= 1;
    }
    return turns;
}

std::string render_prompt(const PromptFrame& frame, const std::vector<Turn>& turns,
                          const std::vector<std::int64_t>& voice_frames) {
    std::string prompt = frame.system_prompt;
    if (!voice_frames.empty()) {
        prompt += frame.voice_input_header;
        for (std::size_t speaker = 0; speaker < voice_frames.size(); ++speaker) {
            prompt += " Speaker " + std::to_string(speaker) + ":" + frame.audio_bos_token;
            for (std::int64_t index = 0; index < voice_frames[speaker]; ++index)
                prompt += frame.audio_token;
            prompt += frame.audio_eos_token + "\n";
        }
    }
    prompt += frame.text_input_header;
    for (const auto& turn : turns)
        prompt += " Speaker " + std::to_string(turn.speaker) + ": " + turn.text + "\n";
    prompt += frame.speech_output_header + frame.audio_bos_token;
    return prompt;
}

} // namespace trtmc::vibevoice
