/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2_moe/runtime/chat_templates.h"

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
    const std::string source = "{{ bos_token }}<|im_start|>{{ message['role'] }}<|im_end|>";
    check(trtmc::lfm2_moe_detect_chat_template_format(source) == "chatml",
          "detects LFM2-MoE ChatML template");
    check(trtmc::lfm2_moe_detect_chat_template_format("plain") == "",
          "does not guess unknown templates");

    const std::string expected = "<|startoftext|><|im_start|>user\nWho are you?<|im_end|>\n"
                                 "<|im_start|>assistant\n";
    check(trtmc::lfm2_moe_apply_chat_template("chatml", "Who are you?") == expected,
          "renders exact official single-user template");
    check(expected.find("system") == std::string::npos, "does not inject a system prompt");
    check(trtmc::lfm2_moe_apply_chat_template("", "raw") == "raw", "leaves raw prompts unchanged");
    const std::string thinking_source = source + "{{- \"<|im_start|>assistant\\n<think>\" -}}";
    check(trtmc::lfm2_moe_detect_chat_template_format(thinking_source) == "chatml_thinking",
          "detects the LFM2-MoE.5 thinking generation prefix");
    check(trtmc::lfm2_moe_apply_chat_template("chatml_thinking", "Who are you?") ==
              expected + "<think>",
          "preserves the LFM2-MoE.5 thinking prefix");
    return failures;
}
