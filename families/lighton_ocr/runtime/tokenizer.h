/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::lighton_ocr {

// Image-only OCR uses a fixed processor-derived prompt. Only output byte-level
// detokenization belongs here; arbitrary text instruction encoding is unsupported.
class Tokenizer {
  public:
    Tokenizer(std::string_view document, std::int32_t vocab_size);
    std::string decode(const std::vector<std::int32_t>& ids) const;

  private:
    std::vector<std::string> bytes_;
    std::vector<bool> special_;
};

} // namespace trtmc::lighton_ocr
