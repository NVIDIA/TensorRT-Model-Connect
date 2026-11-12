/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::moss_transcribe_diarize {

// Bytes a UTF-8 decoder with U+FFFD replacement of each maximal invalid
// subpart yields (Python errors="replace", Rust from_utf8_lossy).
std::string utf8_lossy(std::string_view bytes);

// Python str.strip(): removes leading and trailing Unicode whitespace.
std::string python_strip(std::string_view text);

// Generated ids -> transcript text, as the reference decodes it:
// tokenizer.decode(ids, skip_special_tokens=True).strip().
class TokenDecoder {
  public:
    explicit TokenDecoder(const std::vector<char>& table);
    std::string decode(const std::vector<std::int32_t>& ids) const;
    std::size_t size() const { return flags_.size(); }

  private:
    std::vector<std::uint32_t> offsets_;
    std::vector<std::uint8_t> flags_;
    std::string bytes_;
};

// One "[start][Sxx]text[end]" segment of the model's transcript.
struct TranscriptSegment {
    double start_seconds{0.0};
    double end_seconds{0.0};
    std::string speaker;
    std::string text;
};

// The official streaming parser's result for a complete transcript: segments
// whose end timestamp is not before their start; malformed spans are skipped.
std::vector<TranscriptSegment> parse_transcript(std::string_view text);

} // namespace trtmc::moss_transcribe_diarize
