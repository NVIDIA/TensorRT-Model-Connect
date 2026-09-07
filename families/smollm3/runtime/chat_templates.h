/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ctime>
#include <string>

namespace trtmc {

// Renders "%d <English month> %Y" for a UTC timestamp. The month name is not
// taken from strftime's %B, which follows LC_TIME; taking the parameter keeps
// the formatting testable on any calendar day.
std::string smollm3_format_date(std::time_t when);
std::string smollm3_detect_chat_template_format(const std::string& jinja_template);
std::string smollm3_apply_chat_template(const std::string& format, const std::string& prompt,
                                        bool enable_thinking = true, const std::string& today = {});

} // namespace trtmc
