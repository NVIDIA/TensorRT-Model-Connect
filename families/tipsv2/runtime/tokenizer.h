/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/tipsv2/runtime/charsmap.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace trtmc::tipsv2 {

// SentencePiece ModelProto.SentencePiece.Type, as the build step copies it.
enum class PieceType : std::uint8_t {
    kNormal = 1,
    kUnknown = 2,
    kControl = 3,
    kUserDefined = 4,
    kUnused = 5,
    kByte = 6,
};

struct TokenizerPiece {
    std::string text;
    float score{0.0F};
    PieceType type{PieceType::kNormal};
};

struct LowercaseEntry {
    std::uint32_t code_point{0};
    std::string lowered;
};

// A closed, sorted, non-touching span of code points.
struct CodePointRange {
    std::uint32_t first{0};
    std::uint32_t last{0};
};

// What `families/tipsv2/tokenizer_export.py` flattens `tokenizer.model` into. The runtime links
// neither protobuf nor SentencePiece; everything it needs is here.
//
// Binary layout (little endian):
//   8 bytes magic "TRTMCTIP", u32 version, u32 flags,
//   i32 unknown/bos/eos/pad, u32 piece_count, u32 charsmap_length, u32 lowercase_count,
//   u32 cased_range_count, u32 case_ignorable_range_count,
//   piece records:     u8 type, f32 score, u32 byte_length, raw UTF-8 bytes,
//   charsmap:          raw SentencePiece precompiled charsmap bytes,
//   lowercase records: u32 code point, u32 byte_length, raw UTF-8 bytes,
//   cased ranges:      u32 first, u32 last,
//   ignorable ranges:  u32 first, u32 last.
// Flags are, in order, dummy-prefix, remove-extra-space, escape-space, byte-fallback and
// whitespace-as-suffix.
struct TokenizerAsset {
    static constexpr std::uint32_t kCurrentVersion = 1;

    std::uint32_t version{kCurrentVersion};
    bool add_dummy_prefix{true};
    bool remove_extra_whitespaces{true};
    bool escape_whitespaces{true};
    bool byte_fallback{true};
    bool treat_whitespace_as_suffix{false};
    std::int32_t unknown_id{-1};
    std::int32_t bos_id{-1};
    std::int32_t eos_id{-1};
    std::int32_t pad_id{-1};
    std::vector<TokenizerPiece> pieces;
    std::string charsmap;
    std::vector<LowercaseEntry> lowercase;
    // Unicode's `Cased` and `Case_Ignorable`, which together are the whole of Python's contextual
    // rule for U+03A3: it lowercases to the final form U+03C2 when a cased character precedes it
    // (skipping case-ignorable ones) and none follows, and to U+03C3 otherwise.
    std::vector<CodePointRange> cased;
    std::vector<CodePointRange> case_ignorable;
};

TokenizerAsset parse_tokenizer_asset(std::string_view bytes);

// The checkpoint's tokenizer: Python's `str.lower()`, then SentencePiece's `nmt_nfkc` normalizer,
// then SentencePiece BPE with byte fallback. `text_encoder.py` calls it with BOS and EOS
// explicitly disabled, so `encode` adds neither.
class Tokenizer {
  public:
    explicit Tokenizer(TokenizerAsset asset);

    std::string lowercase(std::string_view text) const;
    std::string normalize(std::string_view text) const;
    std::vector<std::int32_t> encode(std::string_view text) const;

    const TokenizerAsset& asset() const noexcept { return asset_; }
    // The most normalized bytes one token can cover: the longest piece, and never fewer than the
    // four of one UTF-8 character. Used to bound work before tokenizing a long prompt.
    std::size_t longest_piece_bytes() const noexcept { return longest_piece_bytes_; }

  private:
    struct PieceIndex {
        std::int32_t id{-1};
        float score{0.0F};
        PieceType type{PieceType::kNormal};
    };

    std::size_t user_defined_prefix_length(std::string_view text) const;
    std::pair<std::string, std::size_t> normalized_prefix(std::string_view text) const;
    bool is_cased(std::uint32_t code_point) const;
    bool is_case_ignorable(std::uint32_t code_point) const;

    TokenizerAsset asset_;
    PrecompiledCharsMap charsmap_;
    std::unordered_map<std::string, PieceIndex> merge_pieces_;
    std::unordered_map<std::uint32_t, std::string> lowercase_;
    // SentencePiece matches user-defined symbols leftmost-longest. Bucketing them by first byte
    // keeps an ordinary prompt character from scanning the whole table. This checkpoint declares
    // none, but a sibling TIPSv2 release could.
    std::array<std::vector<std::string>, 256> user_defined_by_first_byte_;
    std::array<std::int32_t, 256> byte_ids_{};
    std::size_t longest_piece_bytes_{4};
};

} // namespace trtmc::tipsv2
