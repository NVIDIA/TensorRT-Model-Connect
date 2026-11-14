/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::cosmos3_edge {
class Tokenizer {
  public:
    explicit Tokenizer(std::string_view serialized);
    ~Tokenizer();
    std::vector<int32_t> encode(std::string_view text) const;
    std::string decode(const std::vector<int32_t>& ids) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace trtmc::cosmos3_edge
