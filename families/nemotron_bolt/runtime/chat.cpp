/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "chat.h"

#include <stdexcept>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

namespace trtmc::nemotron_bolt {
namespace {
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
} // namespace

std::string render_chat(const std::vector<ChatMessage>& messages, bool enable_thinking) {
    if (messages.empty())
        throw std::invalid_argument("Bolt chat requires at least one message");
    size_t last_user = messages.size();
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        if (message.role != "user" && message.role != "system" && message.role != "assistant" &&
            message.role != "developer")
            throw std::invalid_argument("Bolt native chat does not expose tool execution");
        if (message.role == "user")
            last_user = i;
    }
    const bool system = messages.front().role == "system";
    std::string result = "<|im_start|>system\n";
    if (system)
        result += messages.front().content;
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
        }
        result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }
    result += "<|im_start|>assistant\n";
    result += enable_thinking ? "<think>\n" : "<think></think>";
    return result;
}
} // namespace trtmc::nemotron_bolt
