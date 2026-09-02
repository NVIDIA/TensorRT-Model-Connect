/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/json_helpers.h"

#include <nlohmann/json.hpp>

namespace trtmc {

namespace {

nlohmann::json parse_json_prefix(const std::string& text) {
    if (text.empty()) {
        return nlohmann::json::object();
    }
    auto it = text.begin();
    nlohmann::json j = nlohmann::json::parse(it, text.end(), nullptr, false, true);
    if (j.is_discarded() || !j.is_object()) {
        return nlohmann::json::object();
    }
    return j;
}

} // namespace

std::string extract_json_string(const std::string& text, const std::string& key,
                                const std::string& fallback) {
    nlohmann::json j = parse_json_prefix(text);
    auto it = j.find(key);
    if (it != j.end() && it->is_string()) {
        return it->get<std::string>();
    }
    return fallback;
}

int32_t extract_json_int(const std::string& text, const std::string& key, int32_t fallback) {
    nlohmann::json j = parse_json_prefix(text);
    auto it = j.find(key);
    if (it != j.end()) {
        if (it->is_number_integer() || it->is_number_unsigned()) {
            return it->get<int32_t>();
        } else if (it->is_number_float()) {
            return static_cast<int32_t>(it->get<double>());
        }
    }
    return fallback;
}

int32_t extract_json_int_or_first_array(const std::string& text, const std::string& key,
                                        int32_t fallback) {
    nlohmann::json j = parse_json_prefix(text);
    auto it = j.find(key);
    if (it != j.end()) {
        if (it->is_number()) {
            return it->is_number_float() ? static_cast<int32_t>(it->get<double>())
                                         : it->get<int32_t>();
        } else if (it->is_array() && !it->empty()) {
            auto first = it->at(0);
            if (first.is_number()) {
                return first.is_number_float() ? static_cast<int32_t>(first.get<double>())
                                               : first.get<int32_t>();
            }
        }
    }
    return fallback;
}

float extract_json_float(const std::string& text, const std::string& key, float fallback) {
    nlohmann::json j = parse_json_prefix(text);
    auto it = j.find(key);
    if (it != j.end() && it->is_number()) {
        return it->get<float>();
    }
    return fallback;
}

std::vector<int32_t> extract_json_int_array(const std::string& text, const std::string& key,
                                            std::size_t max_count) {
    nlohmann::json j = parse_json_prefix(text);
    auto it = j.find(key);
    std::vector<int32_t> out;
    if (it != j.end() && it->is_array()) {
        for (const auto& elem : *it) {
            if (out.size() >= max_count)
                break;
            if (elem.is_number()) {
                out.push_back(elem.is_number_float() ? static_cast<int32_t>(elem.get<double>())
                                                     : elem.get<int32_t>());
            } else {
                break;
            }
        }
    }
    return out;
}

std::vector<std::string> extract_json_string_array(const std::string& text, const std::string& key,
                                                   std::size_t max_count) {
    nlohmann::json j = parse_json_prefix(text);
    auto it = j.find(key);
    std::vector<std::string> out;
    if (it != j.end() && it->is_array()) {
        for (const auto& elem : *it) {
            if (out.size() >= max_count)
                break;
            if (elem.is_string()) {
                out.push_back(elem.get<std::string>());
            } else {
                break;
            }
        }
    }
    return out;
}

bool extract_json_string_array_strict(const std::string& text, const std::string& key,
                                      std::size_t max_count, std::vector<std::string>& out) {
    out.clear();
    nlohmann::json j = parse_json_prefix(text);
    auto it = j.find(key);
    if (it == j.end() || !it->is_array() || it->size() > max_count)
        return false;

    out.reserve(it->size());
    for (const auto& elem : *it) {
        if (!elem.is_string()) {
            out.clear();
            return false;
        }
        out.push_back(elem.get<std::string>());
    }
    return true;
}

std::vector<float> extract_json_float_array(const std::string& text, const std::string& key,
                                            std::size_t max_count) {
    nlohmann::json j = parse_json_prefix(text);
    auto it = j.find(key);
    std::vector<float> out;
    if (it != j.end() && it->is_array()) {
        for (const auto& elem : *it) {
            if (out.size() >= max_count)
                break;
            if (elem.is_number()) {
                out.push_back(elem.get<float>());
            } else {
                break;
            }
        }
    }
    return out;
}

bool extract_json_bool(const std::string& text, const std::string& key, bool fallback) {
    nlohmann::json j = parse_json_prefix(text);
    auto it = j.find(key);
    if (it != j.end() && it->is_boolean()) {
        return it->get<bool>();
    }
    return fallback;
}

std::vector<bool> extract_json_bool_array(const std::string& text, const std::string& key,
                                          std::size_t max_count) {
    nlohmann::json j = parse_json_prefix(text);
    auto it = j.find(key);
    std::vector<bool> out;
    if (it != j.end() && it->is_array()) {
        for (const auto& elem : *it) {
            if (out.size() >= max_count)
                break;
            if (elem.is_boolean()) {
                out.push_back(elem.get<bool>());
            } else {
                break;
            }
        }
    }
    return out;
}

std::string extract_json_object_text(const std::string& text, const std::string& key) {
    nlohmann::json j = parse_json_prefix(text);
    auto it = j.find(key);
    if (it != j.end() && it->is_object()) {
        return it->dump();
    }
    return "";
}

} // namespace trtmc
