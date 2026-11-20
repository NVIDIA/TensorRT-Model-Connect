/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpm_ocr/runtime/prompt.h"

#include <stdexcept>
#include <string_view>

namespace trtmc::minicpm_ocr {
namespace {

// UTF-8 spellings of the characters for which Python's str.isspace() is true.
const std::vector<std::string_view>& whitespace() {
    static const std::vector<std::string_view> spaces{" ",
                                                      "\t",
                                                      "\n",
                                                      "\r",
                                                      "\v",
                                                      "\f",
                                                      "\x1c",
                                                      "\x1d",
                                                      "\x1e",
                                                      "\x1f",
                                                      "\xc2\x85",
                                                      "\xc2\xa0",
                                                      "\xe1\x9a\x80",
                                                      "\xe2\x80\x80",
                                                      "\xe2\x80\x81",
                                                      "\xe2\x80\x82",
                                                      "\xe2\x80\x83",
                                                      "\xe2\x80\x84",
                                                      "\xe2\x80\x85",
                                                      "\xe2\x80\x86",
                                                      "\xe2\x80\x87",
                                                      "\xe2\x80\x88",
                                                      "\xe2\x80\x89",
                                                      "\xe2\x80\x8a",
                                                      "\xe2\x80\xa8",
                                                      "\xe2\x80\xa9",
                                                      "\xe2\x80\xaf",
                                                      "\xe2\x81\x9f",
                                                      "\xe3\x80\x80"};
    return spaces;
}

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string strip_char(std::string text, char character, bool left, bool right) {
    std::size_t begin = 0, end = text.size();
    while (left && begin < end && text[begin] == character)
        ++begin;
    while (right && end > begin && text[end - 1] == character)
        --end;
    return text.substr(begin, end - begin);
}

std::string render_content(const PromptMessage& message, const PromptTokens& tokens) {
    std::string out;
    for (std::size_t i = 0; i < message.parts.size(); ++i) {
        const auto& part = message.parts[i];
        if (i)
            out += '\n';
        if (part.image) {
            if (message.role == PromptRole::System)
                throw std::invalid_argument("System message cannot contain images.");
            out += tokens.image_pad;
        } else {
            if (part.text.find(tokens.image_pad) != std::string::npos ||
                part.text.find("<|video_pad|>") != std::string::npos)
                throw std::invalid_argument(
                    "MiniCPM-OCR text parts cannot spell the image or video placeholder token");
            out += part.text;
        }
    }
    return out;
}

void append_assistant(std::string& out, std::string content, bool after_last_query) {
    std::string reasoning;
    const auto close = content.find("</think>");
    if (close != std::string::npos) {
        // content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
        reasoning = strip_char(content.substr(0, close), '\n', false, true);
        const auto open = reasoning.rfind("<think>");
        if (open != std::string::npos)
            reasoning = reasoning.substr(open + 7);
        reasoning = strip_char(reasoning, '\n', true, false);
        // content.split('</think>')[-1].lstrip('\n')
        content = strip_char(content.substr(content.rfind("</think>") + 8), '\n', true, false);
    }
    reasoning = python_strip(reasoning);
    if (after_last_query && !reasoning.empty())
        out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content;
    else
        out += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + content;
    out += "<|im_end|>\n";
}

} // namespace

std::string python_strip(const std::string& text) {
    std::size_t begin = 0, end = text.size();
    for (bool found = true; found && begin < end;) {
        found = false;
        for (const auto space : whitespace())
            if (starts_with(std::string_view(text).substr(begin, end - begin), space)) {
                begin += space.size();
                found = true;
                break;
            }
    }
    for (bool found = true; found && begin < end;) {
        found = false;
        for (const auto space : whitespace())
            if (ends_with(std::string_view(text).substr(begin, end - begin), space)) {
                end -= space.size();
                found = true;
                break;
            }
    }
    return text.substr(begin, end - begin);
}

std::string replace_invalid_utf8(const std::string& bytes) {
    static constexpr std::string_view replacement{"\xef\xbf\xbd"};
    const auto byte = [&](std::size_t i) { return static_cast<unsigned char>(bytes[i]); };
    std::string out;
    out.reserve(bytes.size());
    for (std::size_t i = 0; i < bytes.size();) {
        const unsigned char lead = byte(i);
        if (lead < 0x80) {
            out += static_cast<char>(lead);
            ++i;
            continue;
        }
        // Well-formed sequences (Unicode Table 3-7): the second byte's range depends on the lead
        // byte.
        std::size_t length = 0;
        unsigned char low = 0x80, high = 0xbf;
        if (lead >= 0xc2 && lead <= 0xdf) {
            length = 2;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            length = 3;
            low = lead == 0xe0 ? 0xa0 : 0x80;
            high = lead == 0xed ? 0x9f : 0xbf;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            length = 4;
            low = lead == 0xf0 ? 0x90 : 0x80;
            high = lead == 0xf4 ? 0x8f : 0xbf;
        }
        std::size_t valid = length ? 1 : 0;
        if (length && i + 1 < bytes.size() && byte(i + 1) >= low && byte(i + 1) <= high) {
            valid = 2;
            while (valid < length && i + valid < bytes.size() && byte(i + valid) >= 0x80 &&
                   byte(i + valid) <= 0xbf)
                ++valid;
        }
        if (length && valid == length) {
            out.append(bytes, i, length);
            i += length;
        } else {
            out += replacement;
            i += valid ? valid : 1;
        }
    }
    return out;
}

std::string render_chat(const std::vector<PromptMessage>& messages, bool enable_thinking,
                        const PromptTokens& tokens) {
    if (messages.empty())
        throw std::invalid_argument("No messages provided.");
    std::vector<std::string> contents;
    contents.reserve(messages.size());
    for (const auto& message : messages)
        contents.push_back(python_strip(render_content(message, tokens)));
    // The last user turn that is not a tool response is the current query.
    int64_t last_query = -1;
    for (int64_t i = static_cast<int64_t>(messages.size()) - 1; i >= 0; --i) {
        const auto& content = contents[static_cast<std::size_t>(i)];
        if (messages[static_cast<std::size_t>(i)].role == PromptRole::User &&
            !(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>"))) {
            last_query = i;
            break;
        }
    }
    if (last_query < 0)
        throw std::invalid_argument("No user query found in messages.");
    std::string out;
    if (messages.front().role == PromptRole::System)
        out += "<|im_start|>system\n" + contents.front() + "<|im_end|>\n";
    for (std::size_t i = 0; i < messages.size(); ++i) {
        switch (messages[i].role) {
        case PromptRole::System:
            if (i != 0)
                throw std::invalid_argument("System message must be at the beginning.");
            break;
        case PromptRole::User:
            out += "<|im_start|>user\n" + contents[i] + "<|im_end|>\n";
            break;
        case PromptRole::Assistant:
            append_assistant(out, contents[i], static_cast<int64_t>(i) > last_query);
            break;
        }
    }
    out += "<|im_start|>assistant\n";
    out += enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return out;
}

std::string expand_images(const std::string& rendered, const std::vector<ImageTokenLayout>& images,
                          bool use_image_id, const PromptTokens& tokens) {
    std::string out;
    std::size_t cursor = 0, index = 0;
    const auto pads = [&](int32_t count) {
        std::string run;
        run.reserve(static_cast<std::size_t>(count) * tokens.image_pad.size());
        for (int32_t i = 0; i < count; ++i)
            run += tokens.image_pad;
        return run;
    };
    for (auto found = rendered.find(tokens.image_pad); found != std::string::npos;
         found = rendered.find(tokens.image_pad, cursor)) {
        if (index >= images.size())
            throw std::invalid_argument(
                "MiniCPM-OCR prompt has more image placeholders than images");
        const auto& image = images[index];
        out.append(rendered, cursor, found - cursor);
        if (use_image_id)
            out += tokens.image_id_start + std::to_string(index) + tokens.image_id_end;
        out += tokens.image_start + pads(image.overview_tokens) + tokens.image_end;
        if (image.grid_rows > 0 && image.grid_cols > 0) {
            if (image.slice_tokens.size() !=
                static_cast<std::size_t>(image.grid_rows) * image.grid_cols)
                throw std::invalid_argument("MiniCPM-OCR slice layout disagrees with its grid");
            for (int32_t row = 0; row < image.grid_rows; ++row) {
                if (row)
                    out += '\n';
                for (int32_t col = 0; col < image.grid_cols; ++col)
                    out += tokens.slice_start +
                           pads(image.slice_tokens[static_cast<std::size_t>(row) * image.grid_cols +
                                                   col]) +
                           tokens.slice_end;
            }
        }
        cursor = found + tokens.image_pad.size();
        ++index;
    }
    if (index != images.size())
        throw std::invalid_argument("MiniCPM-OCR prompt has fewer image placeholders than images");
    out.append(rendered, cursor, std::string::npos);
    return out;
}

} // namespace trtmc::minicpm_ocr
