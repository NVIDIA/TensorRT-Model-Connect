/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace trtmc {

std::string gemma4_generation_suffix(const std::string& format, bool enable_thinking);

std::string gemma_trim_chat_text(const std::string& text);
std::string gemma_detect_chat_template_format(const std::string& jinja_template);
std::string gemma_apply_chat_template(const std::string& format, const std::string& prompt,
                                      bool enable_thinking = true,
                                      const std::string& system_prompt = {});

} // namespace trtmc
