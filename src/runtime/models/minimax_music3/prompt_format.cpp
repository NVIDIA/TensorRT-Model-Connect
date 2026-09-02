/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_music3/prompt_format.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include <vector>

namespace trtmc::minimax_music3 {
namespace {

constexpr const char* kImStart = "<|im_start|>";
constexpr const char* kImEnd = "<|im_end|>";
constexpr const char* kCaptionStart = "<|caption_start|>";
constexpr const char* kCaptionEnd = "<|caption_end|>";
constexpr const char* kLyricsStart = "<|lyrics_start|>";
constexpr const char* kLyricsEnd = "<|lyrics_end|>";
constexpr const char* kAudioStart = "<|audio_start|>";

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (true) {
        const auto stop = text.find('\n', start);
        if (stop == std::string::npos) {
            lines.push_back(text.substr(start));
            break;
        }
        lines.push_back(text.substr(start, stop - start));
        start = stop + 1;
    }
    return lines;
}

std::string join_lines(const std::vector<std::string>& lines) {
    std::ostringstream joined;
    for (std::size_t index = 0; index < lines.size(); ++index) {
        if (index != 0)
            joined << '\n';
        joined << lines[index];
    }
    return joined.str();
}

std::string replace_all(std::string text, const std::string& from, const std::string& to) {
    if (from.empty())
        return text;
    std::size_t at = 0;
    while ((at = text.find(from, at)) != std::string::npos) {
        text.replace(at, from.size(), to);
        at += to.size();
    }
    return text;
}

} // namespace

std::string clean_caption(const std::string& caption) {
    // <|key value|> becomes "key is value"; a span with no space keeps its
    // inner text.
    static const std::regex special_tag(R"(<\|([^|]*)\|>)");
    std::string text;
    auto begin = std::sregex_iterator(caption.begin(), caption.end(), special_tag);
    auto end = std::sregex_iterator();
    std::size_t last = 0;
    for (auto it = begin; it != end; ++it) {
        const auto match = *it;
        text.append(caption, last, static_cast<std::size_t>(match.position()) - last);
        std::string inner = match[1].str();
        const auto first = inner.find_first_not_of(" \t");
        const auto stop = inner.find_last_not_of(" \t");
        inner = first == std::string::npos ? std::string() : inner.substr(first, stop - first + 1);
        const auto space = inner.find_first_of(" \t");
        text += space == std::string::npos
                    ? inner
                    : inner.substr(0, space) + " is " +
                          inner.substr(inner.find_first_not_of(" \t", space));
        last = static_cast<std::size_t>(match.position() + match.length());
    }
    text.append(caption, last, std::string::npos);

    static const std::regex heading(R"(^\s{0,3}#{1,6}\s+)");
    static const std::regex bullet(R"(^\s*[*+-]\s+)");
    static const std::regex bold(R"(\*\*([^*]+)\*\*)");
    static const std::regex italic(R"((^|[^*])\*([^*\n]+)\*($|[^*]))");
    static const std::regex rule(R"(^\s*[-*_]{3,}\s*$)");

    std::vector<std::string> lines;
    for (auto line : split_lines(text)) {
        line = std::regex_replace(line, heading, "");
        line = std::regex_replace(line, bullet, "");
        while (true) {
            const auto updated = std::regex_replace(line, bold, "$1");
            if (updated == line)
                break;
            line = updated;
        }
        line = std::regex_replace(line, italic, "$1$2$3");
        const auto stop = line.find_last_not_of(" \t\r");
        line = stop == std::string::npos ? std::string() : line.substr(0, stop + 1);
        lines.push_back(std::regex_replace(line, rule, ""));
    }

    text = join_lines(lines);
    text = replace_all(text, "\xe2\x80\xa2 ", "");
    text = replace_all(text, "    ", "");
    static const std::regex blank_runs(R"(\n{2,})");
    return std::regex_replace(text, blank_runs, "\n");
}

std::string normalize_lyrics(const std::string& lyrics) {
    // A line that opens with structure tags keeps only those tags; anything
    // sharing the line goes, which is the contract the model card warns about.
    static const std::regex leading_tags(R"(^[ \t]*((?:\[[^\]]+\][ \t]*)+))");
    std::vector<std::string> kept;
    for (const auto& line : split_lines(lyrics)) {
        std::smatch match;
        if (std::regex_search(line, match, leading_tags)) {
            std::string tags = match[1].str();
            const auto stop = tags.find_last_not_of(" \t");
            kept.push_back(stop == std::string::npos ? std::string() : tags.substr(0, stop + 1));
        } else {
            kept.push_back(line);
        }
    }

    std::string text = join_lines(kept);
    text = replace_all(text, "] ", "]\n");
    text = replace_all(text, " [", "\n[");
    text = replace_all(text, " ^ ", "\n");

    static const std::regex tag(R"(\[([^\]]+)\])");
    std::string lowered;
    auto begin = std::sregex_iterator(text.begin(), text.end(), tag);
    auto end = std::sregex_iterator();
    std::size_t last = 0;
    for (auto it = begin; it != end; ++it) {
        const auto match = *it;
        lowered.append(text, last, static_cast<std::size_t>(match.position()) - last);
        std::string inner = match[1].str();
        std::transform(inner.begin(), inner.end(), inner.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        lowered += "[" + inner + "]";
        last = static_cast<std::size_t>(match.position() + match.length());
    }
    lowered.append(text, last, std::string::npos);
    return "[start]\n" + lowered;
}

std::string assemble_prompt(const std::string& caption, const std::string& lyrics) {
    return std::string(kImStart) + kCaptionStart + clean_caption(caption) + kCaptionEnd +
           kLyricsStart + normalize_lyrics(lyrics) + kLyricsEnd + kImEnd + kAudioStart;
}

} // namespace trtmc::minimax_music3
