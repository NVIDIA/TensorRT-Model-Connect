/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/internlm/runtime/chat_templates.h"

#include <iostream>
#include <string>

int main() {
    const std::string jinja =
        "{{ bos_token }}{% for message in messages %}<|im_start|>{{ message.role }}\n"
        "{{ message.content }}<|im_end|>\n{% endfor %}";
    if (trtmc::internlm_detect_chat_template_format(jinja) != "chatml") {
        std::cerr << "FAIL: InternLM ChatML template was not detected\n";
        return 1;
    }

    const auto expected =
        "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n";
    const auto result =
        trtmc::internlm_apply_chat_template("chatml", "The capital of France is", false);
    if (result != expected) {
        std::cerr << "FAIL: InternLM ChatML rendering added unsupported framing\n";
        return 1;
    }
    return 0;
}
