/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

namespace trtmc {

std::string qwen_detect_chat_template_format(const std::string& jinja_template);
std::string qwen_apply_chat_template(const std::string& format, const std::string& prompt,
                                     bool enable_thinking = true,
                                     const std::string& system_prompt = {});
std::string
qwen_apply_conversation_template(const std::string& format,
                                 const std::vector<std::pair<std::string, std::string>>& messages,
                                 bool enable_thinking = true);

} // namespace trtmc
