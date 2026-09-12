/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/span.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace trtmc::internal {

class ConfigError : public std::invalid_argument {
  public:
    explicit ConfigError(const std::string& message) : std::invalid_argument(message) {}
};

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

inline const ConfigField* find_config_field(Span<const ConfigField> fields,
                                            std::string_view name) noexcept {
    for (const auto& field : fields) {
        if (field.name == name)
            return &field;
    }
    return nullptr;
}

// The Core validates declared names and kinds; model-specific ranges and
// combinations remain in the family. Do not insert defaults into supplied.
inline void validate_config(Span<const ConfigField> fields, ConfigView supplied) {
    for (std::size_t index = 0; index < supplied.size(); ++index) {
        const auto& entry = supplied[index];
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (supplied[prior].name == entry.name)
                throw ConfigError("duplicate config: " + std::string(entry.name));
        }
        const auto* declared = find_config_field(fields, entry.name);
        if (declared == nullptr)
            throw ConfigError("unknown config: " + std::string(entry.name));
        if (declared->kind != config_kind(entry.value))
            throw ConfigError("config type mismatch: " + std::string(entry.name));
    }
}

inline bool config_provided(ConfigView supplied, std::string_view name) noexcept {
    for (const auto& entry : supplied) {
        if (entry.name == name)
            return true;
    }
    return false;
}

// Returned strings/lists still borrow supplied or field storage. Absence is
// reserved for declared fields with neither an override nor a fixed default.
template <class T>
std::optional<T> config_get(ConfigView supplied, Span<const ConfigField> fields,
                            std::string_view name) {
    const auto* declared = find_config_field(fields, name);
    if (declared == nullptr)
        throw std::logic_error("read of undeclared config field: " + std::string(name));
    if (declared->kind != config_kind(ConfigValue{T{}}))
        throw std::logic_error("config field read with incorrect type: " + std::string(name));
    const ConfigValue* value = nullptr;
    for (const auto& entry : supplied) {
        if (entry.name != name)
            continue;
        if (value != nullptr)
            throw ConfigError("duplicate config: " + std::string(name));
        value = &entry.value;
    }
    if (value == nullptr && declared->default_value)
        value = &*declared->default_value;
    if (value == nullptr)
        return std::nullopt;
    if (const auto* typed = std::get_if<T>(value))
        return *typed;
    throw ConfigError("config type mismatch: " + std::string(name));
}

} // namespace trtmc::internal
