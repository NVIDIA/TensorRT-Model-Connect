/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/lfm2_vl/runtime/tokenizer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc {

bool lfm2_vl_uses_pinned_split_byte_level_pretokenizer(const char* tokenizer_json,
                                                       std::size_t size);
std::vector<std::string> lfm2_vl_tokenizer_added_tokens(const char* tokenizer_json,
                                                        std::size_t size);

// Model-owned implementation of the pinned LFM2-VL Split regex. The returned
// byte slices are fed independently to the native BPE implementation so merges
// cannot cross the same boundaries enforced by Hugging Face tokenizers.
std::vector<std::string> lfm2_vl_split_pretokens(std::string_view text);

// Added tokens are extracted by longest match before ordinary-text splitting,
// preserving ChatML and tool-token IDs exactly.
std::unique_ptr<ITokenizer> lfm2_vl_wrap_pinned_pretokenizer(std::unique_ptr<ITokenizer> tokenizer,
                                                             std::vector<std::string> added_tokens);

} // namespace trtmc
