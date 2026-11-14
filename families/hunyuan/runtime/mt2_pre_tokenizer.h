/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trtmc::hunyuan {
// Exact native Hy-MT2 three-Split contract; no Edge SDK dependency.
bool is_mt2_sequence(const nlohmann::json& config);
void validate_mt2_tokenizer(const nlohmann::json& config);
std::vector<std::string> split_mt2(const std::string& text);
} // namespace trtmc::hunyuan
