/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/bundle.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace trtmc::qwen3_8_flash_next {
// Materialize exactly one validated table section, not the whole native bundle.
// The existing public reader streams sections but does not expose an absolute
// section range. Do not reparse the shared header to guess a data offset.
class LexicalTable {
  public:
    explicit LexicalTable(const BundleReader& bundle);
    ~LexicalTable();
    LexicalTable(const LexicalTable&) = delete;
    LexicalTable& operator=(const LexicalTable&) = delete;
    std::vector<std::uint8_t> rows(const std::vector<std::int64_t>& ids);

  private:
    std::filesystem::path directory_;
    std::ifstream stream_;
    std::uint64_t bytes_;
    void cleanup() noexcept;
};
} // namespace trtmc::qwen3_8_flash_next
