/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/chat_templates.h"

#include <string>

namespace trtmc {
namespace {

std::string apply_chatml(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (!enable_thinking)
        r += "<think>\n\n</think>\n\n";
    return r;
}

} // namespace

std::string qwen_detect_chat_template_format(const std::string& jinja_template) {
    if (jinja_template.empty())
        return {};
    if (jinja_template.find("<|im_start|>") != std::string::npos)
        return "chatml";
    return {};
}

std::string qwen_apply_chat_template(const std::string& format, const std::string& prompt,
                                     bool enable_thinking) {
    if (format.empty())
        return prompt;
    if (format == "chatml")
        return apply_chatml(prompt, enable_thinking);
    return prompt;
}

} // namespace trtmc
