/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm3/runtime/chat_templates.h"

#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    const std::string source = "{{ bos_token }}<|im_start|>{{ message['role'] }}<|im_end|>{% if "
                               "enable_thinking %}<think>{% endif %}";
    check(trtmc::lfm3_detect_chat_template_format(source) == "chatml",
          "detects LFM3 ChatML template");
    check(trtmc::lfm3_detect_chat_template_format("plain") == "",
          "does not guess unknown templates");

    const std::string expected = "<|startoftext|><|im_start|>user\nWho are you?<|im_end|>\n"
                                 "<|im_start|>assistant\n<think>";
    check(trtmc::lfm3_apply_chat_template("chatml", "Who are you?") == expected,
          "renders exact official single-user template");
    check(trtmc::lfm3_apply_chat_template("chatml", "Who are you?", false) == expected + "</think>",
          "disabled thinking closes the reasoning block");
    check(expected.find("system") == std::string::npos, "does not inject a system prompt");
    check(trtmc::lfm3_apply_chat_template("", "raw") == "raw", "leaves raw prompts unchanged");
    return failures;
}
