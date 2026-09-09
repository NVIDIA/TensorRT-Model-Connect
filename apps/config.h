/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/core.hpp"

#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>

namespace trtmc::app {

// Text flags use raw strings and JSON scalar/list syntax. This is application
// input parsing only; the C API receives typed values, never JSON configuration.
inline ConfigValue parse_config_value(const std::string& text, const ConfigField& field) {
    if (field.kind == ConfigKind::String)
        return text;
    const auto value = nlohmann::json::parse(text, nullptr, false);
    auto invalid = [&]() -> ConfigValue {
        throw std::invalid_argument("config '" + field.name + "' has the wrong value type");
    };
    auto integer = [&](const nlohmann::json& item) -> std::int64_t {
        if (!item.is_number_integer() ||
            (item.is_number_unsigned() &&
             item.get<std::uint64_t>() >
                 static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))) {
            throw std::invalid_argument("config '" + field.name + "' must contain int64 values");
        }
        return item.get<std::int64_t>();
    };
    auto real = [&](const nlohmann::json& item) -> double {
        if (!item.is_number() || !std::isfinite(item.get<double>()))
            throw std::invalid_argument("config '" + field.name + "' must contain finite numbers");
        return item.get<double>();
    };
    switch (field.kind) {
    case ConfigKind::I64:
        return integer(value);
    case ConfigKind::F64:
        return real(value);
    case ConfigKind::Bool:
        if (!value.is_boolean())
            return invalid();
        return value.get<bool>();
    case ConfigKind::I64List: {
        if (!value.is_array())
            return invalid();
        std::vector<std::int64_t> values;
        for (const auto& item : value)
            values.push_back(integer(item));
        return values;
    }
    case ConfigKind::F64List: {
        if (!value.is_array())
            return invalid();
        std::vector<double> values;
        for (const auto& item : value)
            values.push_back(real(item));
        return values;
    }
    case ConfigKind::StringList: {
        if (!value.is_array())
            return invalid();
        std::vector<std::string> values;
        for (const auto& item : value) {
            if (!item.is_string())
                return invalid();
            values.push_back(item.get<std::string>());
        }
        return values;
    }
    case ConfigKind::String:
        break;
    }
    return invalid();
}

// Human-readable benchmark receipts. This does not parse or resolve defaults.
inline nlohmann::json config_value_json(const ConfigValue& value) {
    switch (value.kind()) {
    case ConfigKind::I64:
        return value.get<std::int64_t>();
    case ConfigKind::F64:
        return value.get<double>();
    case ConfigKind::Bool:
        return value.get<bool>();
    case ConfigKind::String:
        return value.get<std::string>();
    case ConfigKind::I64List:
        return value.get<std::vector<std::int64_t>>();
    case ConfigKind::F64List:
        return value.get<std::vector<double>>();
    case ConfigKind::StringList:
        return value.get<std::vector<std::string>>();
    }
    throw std::invalid_argument("unknown config kind");
}

} // namespace trtmc::app
