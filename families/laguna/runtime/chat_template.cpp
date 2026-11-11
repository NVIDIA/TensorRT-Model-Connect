/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/laguna/runtime/chat_template.h"

namespace trtmc::laguna {

namespace {

constexpr std::string_view kWhitespace = " \t\n\r\f\v";

std::string_view rstrip(std::string_view text) {
    const auto end = text.find_last_not_of(kWhitespace);
    return end == std::string_view::npos ? std::string_view{} : text.substr(0, end + 1);
}

} // namespace

std::string render_chat(std::string_view user, std::string_view system_prompt,
                        bool enable_thinking) {
    const std::string_view system = system_prompt.empty() ? kDefaultSystemMessage : system_prompt;
    std::string text = "\xE3\x80\x88|EOS|\xE3\x80\x89"; // 〈|EOS|〉
    const std::string_view stripped = rstrip(system);
    if (!stripped.empty()) {
        text += "<system>\n\n";
        text += stripped;
        text += "\n</system>\n";
    }
    text += "<user>\n";
    text += user;
    text += "\n</user>\n<assistant>\n";
    text += enable_thinking ? "<think>" : "</think>";
    return text;
}

} // namespace trtmc::laguna
