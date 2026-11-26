/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Qwen3-ASR output text: detokenization and the official transcript parsing.
//
// Vocabulary: the bundle's vocab.bin holds the bytes of every token id. Decoding concatenates the
// bytes of the non-special ids and reads them as UTF-8, replacing each maximal invalid subsequence
// with U+FFFD -- tokenizer.decode(ids, skip_special_tokens=True) followed by Python's
// bytes.decode("utf-8", errors="replace").
//
// Transcript: Qwen3ASRProcessor.decode(..., return_format="transcription_only"): strip, drop text
// up to the first "assistant\n", collapse long repetitions (_detect_and_fix_repetitions), and keep
// the text after the first "<asr_text>" marker, stripped. Python string semantics (code points,
// str.strip whitespace) are reproduced on UTF-32.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::qwen3_asr {

class Vocabulary {
  public:
    static Vocabulary parse(const std::vector<char>& data);

    std::string decode(const std::vector<std::int32_t>& token_ids) const;
    std::size_t size() const { return pieces_.size(); }

  private:
    std::vector<std::string> pieces_;
    std::vector<std::uint8_t> flags_;
};

// Python's bytes.decode("utf-8", errors="replace"), as UTF-32.
std::u32string decode_utf8_replace(std::string_view bytes);
std::string encode_utf8(const std::u32string& text);

// _detect_and_fix_repetitions of the official parser.
std::u32string fix_repetitions(const std::u32string& text, std::int32_t threshold);

// Transcript of one decoded generation (transcription_only).
std::string transcription_from_output(const std::string& decoded, const std::string& marker,
                                      std::int32_t repetition_threshold);

} // namespace trtmc::qwen3_asr
