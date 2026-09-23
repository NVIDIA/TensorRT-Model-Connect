/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/chat_templates.h"

#include <iostream>
#include <string>

int main() {
    const std::string base = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n";
    const std::string system = "<|im_start|>system\nNormalize text<|im_end|>\n";
    if (trtmc::qwen_apply_chat_template("chatml", "hello", true) != base ||
        trtmc::qwen_apply_chat_template("chatml", "hello", false, "Normalize text") !=
            system + base + "<think>\n\n</think>\n\n") {
        std::cerr << "Qwen ChatML system prompt rendering failed\n";
        return 1;
    }
    return 0;
}
