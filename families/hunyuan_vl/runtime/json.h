/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace trtmc::hunyuan_vl {
// Parses JSON and rejects an object that repeats a key, which nlohmann::json would resolve to
// the last value without notice.
inline nlohmann::json parse_json(std::string_view text, const char* what) {
    std::vector<std::unordered_set<std::string>> keys;
    bool repeated = false;
    auto document =
        nlohmann::json::parse(text.begin(), text.end(),
                              [&](int, nlohmann::json::parse_event_t event, nlohmann::json& value) {
                                  if (event == nlohmann::json::parse_event_t::object_start)
                                      keys.emplace_back();
                                  else if (event == nlohmann::json::parse_event_t::object_end)
                                      keys.pop_back();
                                  else if (event == nlohmann::json::parse_event_t::key &&
                                           !keys.back().insert(value.get<std::string>()).second)
                                      repeated = true;
                                  return true;
                              });
    if (repeated)
        throw std::invalid_argument(std::string("HunYuanVL ") + what + " repeats a key");
    return document;
}
} // namespace trtmc::hunyuan_vl
