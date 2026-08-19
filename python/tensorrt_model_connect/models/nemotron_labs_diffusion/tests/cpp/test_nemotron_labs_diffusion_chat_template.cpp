/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Model-owned chat-template coverage for Nemotron Labs Diffusion.

#include "chat_templates.h"

#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_detect_nemotron_labs_diffusion() {
    std::string tpl = "{%- set truncate_history_thinking = truncate_history_thinking if "
                      "truncate_history_thinking is defined else True %}"
                      "{%- set enable_thinking = enable_thinking if enable_thinking is defined "
                      "else False %}";
    auto fmt = trtmc::nemotron_labs_diffusion_detect_chat_template_format(tpl);
    check(fmt == "nemotron_labs_diffusion", "nemotron-labs-diffusion detection");
}

static void test_apply_nemotron_labs_diffusion_no_thinking() {
    auto result = trtmc::nemotron_labs_diffusion_apply_chat_template("nemotron_labs_diffusion",
                                                                     "hello", false);
    check(result == "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n"
                    "<|im_start|>assistant\n<think></think>",
          "nemotron-labs-diffusion no-thinking application");
}

int main() {

    test_detect_nemotron_labs_diffusion();
    test_apply_nemotron_labs_diffusion_no_thinking();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All nemotron_labs_diffusion chat_template tests passed.\n";
    return 0;
}
