/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/tokenizer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace trtmc {

// Return true only for the LFM2 tokenizer's decoder shape:
// {"type":"Sequence","decoders":[{"type":"ByteLevel", ...}]}.
bool lfm2_moe_uses_sequence_byte_level_decoder(const char* tokenizer_json, std::size_t size);

// Invert Hugging Face/GPT-2 ByteLevel's byte-to-Unicode alphabet. Every valid
// alphabet code point becomes its original byte; UTF-8 units outside that
// alphabet and malformed byte units are preserved verbatim.
std::string lfm2_moe_decode_gpt2_byte_level(std::string_view encoded);

// Preserve native BPE encoding and special-token behavior while correcting the
// incomplete Sequence[ByteLevel] decode implemented by the shared tokenizer.
std::unique_ptr<ITokenizer> lfm2_moe_wrap_byte_level_decoder(std::unique_ptr<ITokenizer> tokenizer);

} // namespace trtmc
