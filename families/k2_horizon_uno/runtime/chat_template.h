/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

inline constexpr char kK2HorizonUnoPublisherChatTemplateFormat[] = "k2_horizon_uno_publisher_v1";

std::string k2_horizon_uno_apply_chat_template(const std::string& format, const std::string& prompt,
                                               const std::string& reasoning_effort);
void k2_horizon_uno_validate_chat_eos_token_ids(const std::vector<int32_t>& eos_token_ids);

} // namespace trtmc
