/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/k2_horizon_uno/runtime/chat_template.h"

#include <algorithm>
#include <stdexcept>

namespace trtmc {

std::string k2_horizon_uno_apply_chat_template(const std::string& format, const std::string& prompt,
                                               const std::string& reasoning_effort) {
    if (format != kK2HorizonUnoPublisherChatTemplateFormat) {
        throw std::invalid_argument("Unsupported K2-Horizon-Uno chat template format: " + format);
    }
    if (reasoning_effort != "high") {
        throw std::invalid_argument(
            "K2-Horizon-Uno native chat supports only reasoning_effort='high'");
    }
    if (prompt.find("<|ifm|") != std::string::npos || prompt.find("<ifm|") != std::string::npos ||
        prompt.find("</ifm|") != std::string::npos) {
        throw std::invalid_argument(
            "K2-Horizon-Uno chat prompts must not contain publisher protocol markers");
    }

    // The tokenizer owns BOS insertion, so this renderer deliberately omits it.
    return "<|ifm|im_start|>user\n" + prompt +
           "<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n";
}

void k2_horizon_uno_validate_chat_eos_token_ids(const std::vector<int32_t>& eos_token_ids) {
    constexpr int32_t end_of_text_id = 1;
    constexpr int32_t end_of_message_id = 250019;
    if (eos_token_ids.size() != 2 ||
        std::find(eos_token_ids.begin(), eos_token_ids.end(), end_of_text_id) ==
            eos_token_ids.end() ||
        std::find(eos_token_ids.begin(), eos_token_ids.end(), end_of_message_id) ==
            eos_token_ids.end()) {
        throw std::invalid_argument("K2-Horizon-Uno requires publisher EOS token IDs {1, 250019}");
    }
}

} // namespace trtmc
