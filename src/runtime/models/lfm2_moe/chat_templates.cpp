/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lfm2_moe/chat_templates.h"

namespace trtmc {

std::string lfm2_moe_detect_chat_template_format(const std::string& jinja_template) {
    if (jinja_template.find("<|im_start|>") != std::string::npos &&
        jinja_template.find("<|im_end|>") != std::string::npos) {
        return "chatml";
    }
    return {};
}

std::string lfm2_moe_apply_chat_template(const std::string& format, const std::string& prompt) {
    if (format != "chatml")
        return prompt;
    // The source template supplies bos_token before the first role. Native
    // tokenizer framing supplies the same BOS; encode_prompt removes one copy
    // after tokenization so the rendered token sequence is exact.
    return "<|startoftext|><|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

} // namespace trtmc
