/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/k2_horizon/chat_template.h"

#include "utils/sha256.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace trtmc {

std::string k2_horizon_detect_chat_template_format(const std::string& jinja_template) {
    if (jinja_template.empty())
        throw std::invalid_argument("K2-Horizon chat template must not be empty");

    internal::Sha256 hash;
    hash.update(jinja_template);
    if (hash.hex_digest() != kK2HorizonPublisherChatTemplateSha256) {
        throw std::invalid_argument(
            "K2-Horizon supports only the pinned publisher chat template contract");
    }
    return kK2HorizonPublisherChatTemplateFormat;
}

std::string k2_horizon_apply_chat_template(const std::string& format, const std::string& prompt,
                                           const std::string& reasoning_effort) {
    if (format.empty())
        throw std::invalid_argument("K2-Horizon chat template format must not be empty");
    if (format != kK2HorizonPublisherChatTemplateFormat) {
        throw std::invalid_argument("Unsupported K2-Horizon chat template format: " + format);
    }
    if (reasoning_effort != "high") {
        throw std::invalid_argument(
            "K2-Horizon native chat currently supports only reasoning_effort='high'");
    }
    if (prompt.find("<|ifm|") != std::string::npos || prompt.find("<ifm|") != std::string::npos ||
        prompt.find("</ifm|") != std::string::npos) {
        throw std::invalid_argument(
            "K2-Horizon chat prompts must not contain publisher protocol markers");
    }

    // The tokenizer owns BOS insertion, so the native renderer deliberately omits bos_token.
    return "<|ifm|im_start|>user\n" + prompt +
           "<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n";
}

void k2_horizon_validate_chat_eos_token_ids(const std::vector<int32_t>& eos_token_ids) {
    constexpr int32_t end_of_text_id = 1;
    constexpr int32_t end_of_message_id = 250019;
    if (eos_token_ids.size() != 2 ||
        std::find(eos_token_ids.begin(), eos_token_ids.end(), end_of_text_id) ==
            eos_token_ids.end() ||
        std::find(eos_token_ids.begin(), eos_token_ids.end(), end_of_message_id) ==
            eos_token_ids.end()) {
        throw std::invalid_argument("K2-Horizon requires publisher EOS token IDs {1, 250019}");
    }
}

} // namespace trtmc
