/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm3/runtime/chat_templates.h"

namespace trtmc {

std::string lfm3_detect_chat_template_format(const std::string& jinja_template) {
    if (jinja_template.find("<|im_start|>") != std::string::npos &&
        jinja_template.find("<|im_end|>") != std::string::npos &&
        jinja_template.find("enable_thinking") != std::string::npos) {
        return "chatml";
    }
    return {};
}

std::string lfm3_apply_chat_template(const std::string& format, const std::string& prompt,
                                     bool enable_thinking) {
    if (format != "chatml")
        return prompt;
    return "<|startoftext|><|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n" +
           std::string(enable_thinking ? "<think>" : "<think></think>");
}

} // namespace trtmc
