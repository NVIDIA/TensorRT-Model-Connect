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

struct SpokenLine {
    std::int32_t speaker;
    std::string_view text;
};

// ``Speaker <digits> : <text>`` with case-insensitive "Speaker" and optional spaces.
std::optional<SpokenLine> speaker_line(std::string_view line) {
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
    bool too_large = false;
    while (cursor < line.size() && std::isdigit(static_cast<unsigned char>(line[cursor])) != 0) {
        // The value stops growing past the int32 range; whether the line declares a
        // speaker, and so has an id too large, is known only at the colon.
        if (!too_large) {
            speaker = speaker * 10 + (line[cursor] - '0');
            too_large = speaker > std::numeric_limits<std::int32_t>::max();
        }
        ++cursor;
    }
    if (cursor == digits)
        return std::nullopt;
    while (cursor < line.size() && is_space(line[cursor]))
        ++cursor;
    if (cursor >= line.size() || line[cursor] != ':')
        return std::nullopt;
    if (too_large)
        throw std::invalid_argument("vibevoice speaker id is too large");
    return SpokenLine{static_cast<std::int32_t>(speaker), trim(line.substr(cursor + 1))};
}

[[noreturn]] void too_long(std::size_t max_text_bytes) {
    throw std::length_error("vibevoice text holds more than " + std::to_string(max_text_bytes) +
                            " bytes to speak");
}

[[noreturn]] void too_many_turns(std::size_t max_turns) {
    throw std::length_error("vibevoice script has more than " + std::to_string(max_turns) +
                            " turns");
}

} // namespace

std::vector<Turn> parse_script(std::string_view text, std::size_t max_text_bytes,
                               std::size_t max_turns) {
    const auto whole = trim(text);
    if (whole.empty())
        throw std::invalid_argument("vibevoice requires non-empty text");
    // One pass over the lines: only views of the spoken text are kept, and the text a script
    // keeps is counted as it is found, so a text that cannot fit is refused before any copy.
    std::vector<SpokenLine> spoken;
    bool plain = false;
    std::size_t kept = 0;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = std::min(text.find('\n', start), text.size());
        const auto line = trim(text.substr(start, end - start));
        start = end + 1;
        if (line.empty())
            continue;
        if (auto parsed = speaker_line(line)) {
            if (parsed->text.empty())
                throw std::invalid_argument("vibevoice script has a speaker line without text");
            kept += parsed->text.size();
            if (kept > max_text_bytes)
                too_long(max_text_bytes);
            if (spoken.size() == max_turns)
                too_many_turns(max_turns);
            spoken.push_back(*parsed);
        } else {
            plain = true;
        }
        if (plain && !spoken.empty())
            throw std::invalid_argument(
                "vibevoice scripts need every non-empty line to read 'Speaker <id>: <text>'");
    }
    if (spoken.empty()) {
        // Plain text is one turn, under the same limits as a script.
        if (whole.size() > max_text_bytes)
            too_long(max_text_bytes);
        if (max_turns == 0)
            too_many_turns(max_turns);
        return {Turn{0, std::string(whole)}};
    }
    const auto lowest = std::min_element(spoken.begin(), spoken.end(),
                                         [](const SpokenLine& a, const SpokenLine& b) {
                                             return a.speaker < b.speaker;
                                         })
                            ->speaker;
    // Ids written from 1 are renumbered from 0.
    const std::int32_t shift = lowest > 0 ? 1 : 0;
    std::vector<Turn> turns;
    turns.reserve(spoken.size());
    for (const auto& line : spoken)
        turns.push_back(Turn{line.speaker - shift, std::string(line.text)});
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
