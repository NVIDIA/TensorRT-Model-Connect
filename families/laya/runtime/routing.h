/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/laya/runtime/record.h"

#include <array>
#include <unordered_map>
#include <unordered_set>

namespace trtmc::laya {
class Routing {
  public:
    explicit Routing(const Json& tables);
    Json route(const Json& document) const;

  private:
    int flags(char32_t code) const;
    std::u32string lower(const std::u32string& text) const;
    std::string script(char32_t code, bool profile) const;
    Json analyse(const Json& state) const;
    std::string name(std::string value) const;
    std::string quoted(const std::string& value) const;
    std::vector<std::array<std::uint32_t, 3>> properties_;
    std::unordered_map<char32_t, std::u32string> lowercase_;
    std::vector<std::pair<std::string, std::unordered_set<std::string>>> stopwords_;
    std::unordered_set<std::string> shared_;
    std::unordered_set<char32_t> diacritics_;
    Json scripts_, aliases_, workflows_;
};
} // namespace trtmc::laya
