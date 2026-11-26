/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace trtmc::bailing_hybrid {
inline int64_t generation_limit(std::optional<int64_t> requested, std::size_t prefix,
                                int32_t profile) {
    if (profile <= 0 || prefix == 0 || prefix >= static_cast<std::size_t>(profile))
        throw std::invalid_argument("Ling3 prefix must leave room in the built KV profile");
    const auto remaining = static_cast<int64_t>(profile) - static_cast<int64_t>(prefix);
    const auto limit = requested.value_or(std::min<int64_t>(32, remaining));
    if (limit <= 0 || limit > remaining)
        throw std::invalid_argument("Ling3 generation exceeds the remaining KV profile");
    return limit;
}
} // namespace trtmc::bailing_hybrid
