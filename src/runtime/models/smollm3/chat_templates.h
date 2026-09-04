/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace trtmc {

std::string smollm3_detect_chat_template_format(const std::string& jinja_template);
std::string smollm3_apply_chat_template(const std::string& format, const std::string& prompt,
                                      bool enable_thinking = true);

} // namespace trtmc
