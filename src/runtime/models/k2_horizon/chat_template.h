/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

inline constexpr char kK2HorizonPublisherChatTemplateFormat[] = "k2_horizon_publisher_v1";
inline constexpr char kK2HorizonPublisherChatTemplateSha256[] =
    "a892cd0b0195599f283a8c706787520d9a6747640efb2f4dec4144b0abb62590";

// Validate the exact bundled publisher Jinja contract and return its native format identifier.
// The runtime never evaluates Jinja; unknown template bytes are rejected fail-closed.
std::string k2_horizon_detect_chat_template_format(const std::string& jinja_template);

// Render the qualified single-user, add-generation-prompt form. The initial native contract
// intentionally supports only the publisher's high reasoning mode.
std::string k2_horizon_apply_chat_template(const std::string& format, const std::string& prompt,
                                           const std::string& reasoning_effort);

// Preserve both publisher stop conditions for native chat generation.
void k2_horizon_validate_chat_eos_token_ids(const std::vector<int32_t>& eos_token_ids);

} // namespace trtmc
