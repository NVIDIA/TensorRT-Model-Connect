/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::phonon {
class Tokenizer {
  public:
    explicit Tokenizer(const std::string& json);
    std::vector<std::int32_t> encode(std::string_view text) const;
    static std::string prepare(std::string_view text);

  private:
    std::vector<std::pair<std::string, double>> vocab_;
};
} // namespace trtmc::phonon
