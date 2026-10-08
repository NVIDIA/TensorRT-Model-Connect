/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hunyuan/runtime/chat_templates.h"

#include <stdexcept>
namespace trtmc {
std::string hunyuan_detect_chat_template_format(const std::string& value) {
    if (value.find("<｜hy_User｜>") != std::string::npos &&
        value.find("<｜hy_Assistant｜>") != std::string::npos)
        return "hunyuan_mt2";
    if (value.find("<|startoftext|>") == std::string::npos ||
        value.find("<|extra_0|>") == std::string::npos)
        throw std::invalid_argument("Unsupported Hunyuan chat template");
    return "hunyuan";
}
std::string hunyuan_apply_chat_template(const std::string& format, const std::string& prompt,
                                        bool) {
    if (format == "hunyuan_mt2")
        return "<｜hy_begin▁of▁sentence｜><｜hy_User｜>" + prompt + "<｜hy_Assistant｜>";
    if (format != "hunyuan")
        throw std::invalid_argument("Unsupported Hunyuan chat template");
    // The public text Task accepts one user message. These checkpoints do not
    // append an assistant prefix when add_generation_prompt is true.
    return "<|startoftext|>" + prompt + "<|extra_0|>";
}
} // namespace trtmc
