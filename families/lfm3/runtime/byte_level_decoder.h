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

namespace trtmc {

bool lfm3_uses_sequence_byte_level_decoder(const char* tokenizer_json, std::size_t size);

std::string lfm3_decode_gpt2_byte_level(std::string_view encoded);

std::unique_ptr<ITokenizer> lfm3_wrap_byte_level_decoder(std::unique_ptr<ITokenizer> tokenizer);

} // namespace trtmc
