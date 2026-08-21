/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/json_helpers.h"
#include <nlohmann/json.hpp>

namespace trtmc {

namespace {
// Parse the input string into a JSON object safely, ignoring comments and without throwing exceptions on parse errors.
// Note: If the JSON is invalid, it returns a discarded object (which `.is_discarded()` will be true).
nlohmann::json parse_json(const std::string& text) {
    return nlohmann::json::parse(text, nullptr, false, true);
}

// Find a value in a parsed JSON document.
// Because nlohmann::json is being parsed as a whole document, this only searches the top-level.
const nlohmann::json* find_value(const nlohmann::json& doc, const std::string& key) {
    if (!doc.is_object()) {
        return nullptr;
    }
    auto it = doc.find(key);
    if (it != doc.end()) {
        return &(*it);
    }
    return nullptr;
}
} // namespace

std::string extract_json_string(const std::string& text, const std::string& key,
                                const std::string& fallback) {
    auto doc = parse_json(text);
    if (auto val = find_value(doc, key)) {
        if (val->is_string()) {
            return val->get<std::string>();
        }
    }
    return fallback;
}

std::vector<std::string> extract_json_string_array(const std::string& text, const std::string& key) {
    std::vector<std::string> result;
    auto doc = parse_json(text);
    if (auto val = find_value(doc, key)) {
        if (val->is_array()) {
            for (const auto& item : *val) {
                if (item.is_string()) {
                    result.push_back(item.get<std::string>());
                }
            }
        }
    }
    return result;
}

int32_t extract_json_int(const std::string& text, const std::string& key, int32_t fallback) {
    auto doc = parse_json(text);
    if (auto val = find_value(doc, key)) {
        if (val->is_number_integer()) {
            return val->get<int32_t>();
        }
        if (val->is_number_float()) {
            return static_cast<int32_t>(val->get<double>());
        }
    }
    return fallback;
}

int32_t extract_json_int_or_first_array(const std::string& text, const std::string& key,
                                        int32_t fallback) {
    auto doc = parse_json(text);
    if (auto val = find_value(doc, key)) {
        if (val->is_number_integer()) {
            return val->get<int32_t>();
        }
        if (val->is_number_float()) {
            return static_cast<int32_t>(val->get<double>());
        }
        if (val->is_array() && !val->empty()) {
            const auto& first = (*val)[0];
            if (first.is_number_integer()) {
                return first.get<int32_t>();
            }
            if (first.is_number_float()) {
                return static_cast<int32_t>(first.get<double>());
            }
        }
    }
    return fallback;
}

float extract_json_float(const std::string& text, const std::string& key, float fallback) {
    auto doc = parse_json(text);
    if (auto val = find_value(doc, key)) {
        if (val->is_number()) {
            return val->get<float>();
        }
    }
    return fallback;
}

std::vector<float> extract_json_float_array(const std::string& text, const std::string& key,
                                            std::size_t max_count) {
    std::vector<float> result;
    auto doc = parse_json(text);
    if (auto val = find_value(doc, key)) {
        if (val->is_array()) {
            for (const auto& item : *val) {
                if (result.size() >= max_count) break;
                if (item.is_number()) {
                    result.push_back(item.get<float>());
                }
            }
        }
    }
    return result;
}

std::vector<int32_t> extract_json_int_array(const std::string& text, const std::string& key,
                                            std::size_t max_count) {
    std::vector<int32_t> result;
    auto doc = parse_json(text);
    if (auto val = find_value(doc, key)) {
        if (val->is_array()) {
            for (const auto& item : *val) {
                if (result.size() >= max_count) break;
                if (item.is_number_integer()) {
                    result.push_back(item.get<int32_t>());
                } else if (item.is_number_float()) {
                    result.push_back(static_cast<int32_t>(item.get<double>()));
                }
            }
        }
    }
    return result;
}

bool extract_json_bool(const std::string& text, const std::string& key, bool fallback) {
    auto doc = parse_json(text);
    if (auto val = find_value(doc, key)) {
        if (val->is_boolean()) {
            return val->get<bool>();
        }
        if (val->is_number_integer()) {
            return val->get<int>() != 0;
        }
    }
    return fallback;
}

std::vector<bool> extract_json_bool_array(const std::string& text, const std::string& key,
                                          std::size_t max_count) {
    std::vector<bool> result;
    auto doc = parse_json(text);
    if (auto val = find_value(doc, key)) {
        if (val->is_array()) {
            for (const auto& item : *val) {
                if (result.size() >= max_count) break;
                if (item.is_boolean()) {
                    result.push_back(item.get<bool>());
                } else if (item.is_number_integer()) {
                    result.push_back(item.get<int>() != 0);
                }
            }
        }
    }
    return result;
}

std::string extract_json_object_text(const std::string& text, const std::string& key) {
    auto doc = parse_json(text);
    if (auto val = find_value(doc, key)) {
        if (val->is_object()) {
            return val->dump();
        }
    }
    return "";
}

} // namespace trtmc
