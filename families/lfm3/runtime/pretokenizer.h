/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/lfm3/runtime/tokenizer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc {

bool lfm3_uses_pinned_split_byte_level_pretokenizer(const char* tokenizer_json, std::size_t size);
std::vector<std::string> lfm3_tokenizer_added_tokens(const char* tokenizer_json, std::size_t size);

std::vector<std::string> lfm3_split_pretokens(std::string_view text);

std::unique_ptr<ITokenizer> lfm3_wrap_pinned_pretokenizer(std::unique_ptr<ITokenizer> tokenizer,
                                                          std::vector<std::string> added_tokens);

} // namespace trtmc
