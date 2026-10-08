/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/chat_templates.h"

#include <string>

namespace trtmc {
namespace {

std::string apply_chatml(const std::string& prompt, bool append_empty_think,
                         const std::string& system_prompt) {
    std::string r;
    if (!system_prompt.empty())
        r = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    r += "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (append_empty_think)
        r += "<think>\n\n</think>\n\n";
    return r;
}

} // namespace

std::string qwen_detect_chat_template_format(const std::string& jinja_template) {
    if (jinja_template.empty())
        return {};
    if (jinja_template.find("<|im_start|>") != std::string::npos) {
        // ChatML framing alone does not imply the hybrid Qwen3 thinking suffix.
        const bool has_empty_think =
            jinja_template.find("<think>\\n\\n</think>\\n\\n") != std::string::npos ||
            jinja_template.find("<think>\n\n</think>\n\n") != std::string::npos;
        if (jinja_template.find("enable_thinking") != std::string::npos && has_empty_think)
            return "chatml_thinking";
        return "chatml";
    }
    return {};
}

std::string qwen_apply_chat_template(const std::string& format, const std::string& prompt,
                                     bool enable_thinking, const std::string& system_prompt) {
    if (format.empty())
        return prompt;
    if (format == "chatml" || format == "chatml_thinking")
        return apply_chatml(prompt, format == "chatml_thinking" && !enable_thinking, system_prompt);
    return prompt;
}

} // namespace trtmc
