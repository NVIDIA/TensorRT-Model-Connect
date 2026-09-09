/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/span.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace trtmc::internal {

// These values borrow all strings and arrays. Copying a value does not extend
// the storage lifetime. A family must own any data retained after a call.
using ConfigValue =
    std::variant<std::int64_t, double, bool, std::string_view, Span<const std::int64_t>,
                 Span<const double>, Span<const std::string_view>>;

// This order follows ConfigValue's alternatives. It is an internal
// discriminator, not the public C ABI's wire-kind numbering.
enum class ConfigKind : std::uint32_t {
    I64,
    F64,
    Bool,
    String,
    I64List,
    F64List,
    StringList,
};

inline ConfigKind config_kind(const ConfigValue& value) noexcept {
    return static_cast<ConfigKind>(value.index());
}

template <typename T>
const T& config_value_as(const ConfigValue& value) {
    if (const auto* typed = std::get_if<T>(&value))
        return *typed;
    throw std::invalid_argument("config value type mismatch");
}

struct ConfigEntry {
    std::string_view name;
    ConfigValue value;
};

// An ordered view, not a map: duplicate names survive transport so the family
// can reject them. An empty view means no explicit overrides.
using ConfigView = Span<const ConfigEntry>;

struct ConfigField {
    std::string_view name;
    ConfigKind kind;
    // Engaged means a fixed default, including false, zero or an empty value.
    // Disengaged means the family computes the default from its context/input.
    std::optional<ConfigValue> default_value;
    std::string_view description;
};

} // namespace trtmc::internal
