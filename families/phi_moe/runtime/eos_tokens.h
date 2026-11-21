/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace trtmc {

inline std::vector<int32_t> phi_moe_parse_eos_tokens(const nlohmann::json& value) {
    if (value.is_number_integer())
        return {value.get<int32_t>()};
    if (value.is_array() && !value.empty() &&
        std::all_of(value.begin(), value.end(),
                    [](const auto& token) { return token.is_number_integer(); }))
        return value.get<std::vector<int32_t>>();
    throw std::runtime_error("phi_moe runtime.json has invalid 'eos_token_id'");
}

inline bool phi_moe_is_eos(int32_t token, int32_t default_eos,
                           const std::vector<int32_t>& checkpoint_ids, int32_t override_eos = -1) {
    if (override_eos >= 0)
        return token == override_eos;
    return token == default_eos ||
           std::find(checkpoint_ids.begin(), checkpoint_ids.end(), token) != checkpoint_ids.end();
}

} // namespace trtmc
