/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_labs_diffusion/runtime/chat_templates.h"

#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_detect_template() {
    const std::string source =
        "{%- set truncate_history_thinking = truncate_history_thinking if "
        "truncate_history_thinking is defined else True %}"
        "{%- set enable_thinking = enable_thinking if enable_thinking is defined else False %}";
    check(trtmc::nemotron_labs_diffusion_detect_chat_template_format(source) ==
              "nemotron_labs_diffusion",
          "detect Nemotron Labs Diffusion template");
}

void test_apply_without_thinking() {
    const auto result = trtmc::nemotron_labs_diffusion_apply_chat_template(
        "nemotron_labs_diffusion", "hello", false);
    check(result == "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n"
                    "<|im_start|>assistant\n<think></think>",
          "apply template without thinking");
}

} // namespace

int main() {
    test_detect_template();
    test_apply_without_thinking();
    return failures;
}
