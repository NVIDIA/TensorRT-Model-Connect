/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace trtmc::phi {

inline std::vector<std::int32_t> require_eos_tokens(const nlohmann::json& json,
                                                    std::int32_t vocab_size) {
    if (!json.contains("eos_token_id"))
        throw std::runtime_error("phi runtime.json missing 'eos_token_id'");
    const auto& value = json.at("eos_token_id");
    std::vector<std::int32_t> ids;
    const auto append = [&](const nlohmann::json& token) {
        if (!token.is_number_integer() || vocab_size <= 0)
            throw std::runtime_error("phi runtime.json has invalid 'eos_token_id'");
        if (token.is_number_unsigned()) {
            if (token.get<std::uint64_t>() >= static_cast<std::uint64_t>(vocab_size))
                throw std::runtime_error("phi runtime.json has out-of-range 'eos_token_id'");
        } else if (token.get<std::int64_t>() < 0 || token.get<std::int64_t>() >= vocab_size) {
            throw std::runtime_error("phi runtime.json has out-of-range 'eos_token_id'");
        }
        ids.push_back(token.get<std::int32_t>());
    };
    if (value.is_number_integer()) {
        append(value);
    } else if (value.is_array() && !value.empty()) {
        for (const auto& token : value)
            append(token);
    } else {
        throw std::runtime_error("phi runtime.json has invalid 'eos_token_id'");
    }
    return ids;
}

} // namespace trtmc::phi
