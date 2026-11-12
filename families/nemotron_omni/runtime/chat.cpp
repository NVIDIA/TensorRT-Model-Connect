/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "chat.h"

#include <stdexcept>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

namespace trtmc::nemotron_omni {
namespace {
void replace(std::string& value, const std::string& from, const std::string& to) {
    size_t position = 0;
    while ((position = value.find(from, position)) != std::string::npos) {
        value.replace(position, from.size(), to);
        position += to.size();
    }
}
bool space(UChar32 value) {
    return u_isUWhiteSpace(value) || (value >= 0x1c && value <= 0x1f);
}
std::string trim(const std::string& value) {
    auto text = icu::UnicodeString::fromUTF8(value);
    int32_t begin = 0, end = text.length();
    while (begin < end && space(text.char32At(begin)))
        begin = text.moveIndex32(begin, 1);
    while (end > begin && space(text.char32At(text.moveIndex32(end, -1))))
        end = text.moveIndex32(end, -1);
    std::string result;
    text.tempSubStringBetween(begin, end).toUTF8String(result);
    return result;
}
std::string sanitize(std::string text) {
    replace(text, "</think>", "<_end_think>");
    replace(text, "/think", "");
    replace(text, "/no_think", "");
    replace(text, "<_end_think>", "</think>");
    return text;
}
} // namespace

std::string render_chat(const std::vector<ChatMessage>& messages, bool enable_thinking) {
    if (messages.empty())
        throw std::invalid_argument("Omni chat requires at least one message");
    size_t last_user = messages.size();
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        if (message.role != "user" && message.role != "system" && message.role != "assistant" &&
            message.role != "developer")
            throw std::invalid_argument("Omni native chat does not expose tool execution");
        if (message.role == "user")
            last_user = i;
        if (message.role == "user" || message.role == "system") {
            auto text = message.content;
            replace(text, "</think>", "");
            if (text.find("/think") != std::string::npos)
                enable_thinking = true;
            else if (text.find("/no_think") != std::string::npos)
                enable_thinking = false;
        }
    }
    const bool system = messages.front().role == "system";
    std::string result = "<|im_start|>system\n";
    if (system)
        result += sanitize(messages.front().content);
    result += "<|im_end|>\n";
    for (size_t index = system ? 1 : 0; index < messages.size(); ++index) {
        auto content = messages[index].content;
        const auto& role = messages[index].role;
        if (role == "assistant") {
            if (content.find("<think>") == std::string::npos &&
                content.find("</think>") == std::string::npos)
                content = "<think></think>" + content;
            if (last_user != messages.size() && index < last_user &&
                content.find("<think>") != std::string::npos &&
                content.find("</think>") != std::string::npos)
                content = "<think></think>" + content.substr(content.rfind("</think>") + 8);
            content = trim(content);
        } else if (role == "user" || role == "system")
            content = trim(sanitize(content));
        result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }
    result += "<|im_start|>assistant\n";
    result += enable_thinking ? "<think>\n" : "<think></think>";
    return result;
}
} // namespace trtmc::nemotron_omni
