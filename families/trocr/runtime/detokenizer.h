/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/span.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::trocr {

// Turns generated ids into text the way the checkpoint's RobertaTokenizer
// decodes with skip_special_tokens: special ids are dropped, every other token
// is mapped back through GPT-2's byte-to-unicode table, and the bytes are read
// as UTF-8 with each ill-formed subsequence replaced by U+FFFD.
class ByteLevelDetokenizer {
  public:
    ByteLevelDetokenizer(std::vector<std::string> id_to_token,
                         const std::vector<std::int32_t>& special_ids);

    std::string decode(Span<const std::int32_t> ids) const;
    std::size_t size() const noexcept { return token_bytes_.size(); }

  private:
    std::vector<std::string> token_bytes_;
    std::vector<bool> special_;
};

// UTF-8 with every maximal ill-formed subpart replaced by U+FFFD.
std::string utf8_replace_invalid(std::string_view bytes);

} // namespace trtmc::trocr
