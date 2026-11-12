/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trtmc::paligemma {

// PaliGemma uses the SentencePiece BPE model, not SentencePiece's Unigram
// model. The build-time converter flattens protobuf-only data into this runtime
// representation so inference does not link protobuf or SentencePiece.
enum class SentencePieceType : uint8_t {
    kNormal = 1,
    kUnknown = 2,
    kControl = 3,
    kUserDefined = 4,
    kUnused = 5,
    kByte = 6,
};

struct BpePiece {
    std::string text;
    float score{0.0F};
    SentencePieceType type{SentencePieceType::kNormal};
};

struct NormalizationRule {
    std::string source;
    std::string replacement;
};

struct PaligemmaBpeAsset {
    static constexpr uint32_t kCurrentVersion = 1;

    uint32_t version{kCurrentVersion};
    bool add_dummy_prefix{true};
    bool remove_extra_whitespaces{true};
    bool escape_whitespaces{true};
    bool byte_fallback{true};
    bool treat_whitespace_as_suffix{false};
    int32_t unknown_id{-1};
    int32_t bos_id{-1};
    int32_t eos_id{-1};
    int32_t pad_id{-1};
    std::vector<BpePiece> pieces;
    // Leftmost-longest rules expanded from NormalizerSpec.precompiled_charsmap.
    std::vector<NormalizationRule> normalization_rules;
};

// Binary layout (little endian):
//   8 bytes magic "TRTMCBPE", u32 version, u32 flags,
//   i32 unk/bos/eos/pad, u32 piece_count, u32 normalization_rule_count,
//   piece records: u8 type, f32 score, u32 byte_length, raw UTF-8 bytes,
//   rule records: u32 source_length, u32 replacement_length, raw bytes.
// Flags are, in order, dummy-prefix, remove-extra-space, escape-space,
// byte-fallback, and whitespace-as-suffix. This deliberately is not a
// SentencePiece protobuf parser.
PaligemmaBpeAsset parse_paligemma_bpe_asset(std::string_view bytes);
std::vector<uint8_t> serialize_paligemma_bpe_asset(const PaligemmaBpeAsset& asset);

class PaligemmaBpeTokenizer {
  public:
    explicit PaligemmaBpeTokenizer(PaligemmaBpeAsset asset);

    std::string normalize(std::string_view text) const;
    std::vector<int32_t> encode(std::string_view text, bool add_bos) const;

    // Mirrors the reference decoder (skip, U+2581 -> space, byte fallback,
    // fuse). Runs of byte pieces that are not valid UTF-8 become one U+FFFD per
    // byte. Ids outside the vocabulary have no text.
    std::string decode(const std::vector<int32_t>& ids,
                       const std::unordered_set<int32_t>& skipped) const;
    const PaligemmaBpeAsset& asset() const noexcept { return asset_; }
    // Most normalized-text bytes one token covers: the longest piece, and at
    // least one UTF-8 character (an unknown piece without byte fallback).
    std::size_t longest_piece_bytes() const noexcept { return longest_piece_bytes_; }
    // True when normalization can only lengthen text (no whitespace removal, no
    // shortening rule), so the raw byte length bounds the normalized one.
    bool normalization_never_shortens() const noexcept { return never_shortens_; }

  private:
    struct PieceIndex {
        int32_t id{-1};
        float score{0.0F};
        SentencePieceType type{SentencePieceType::kNormal};
    };

    std::size_t user_defined_prefix_length(std::string_view text) const;
    std::pair<std::string, std::size_t> normalized_prefix(std::string_view text) const;

    PaligemmaBpeAsset asset_;
    std::unordered_map<std::string, PieceIndex> merge_pieces_;
    // SentencePiece user-defined symbols are matched leftmost-longest. Bucket
    // them by their first byte so ordinary prompt characters do not scan the
    // full (1,397-entry in the released PaliGemma asset) symbol table.
    std::array<std::vector<std::string>, 256> user_defined_pieces_by_first_byte_;
    std::vector<NormalizationRule> normalization_rules_;
    std::array<int32_t, 256> byte_ids_{};
    std::size_t longest_piece_bytes_{4};
    bool never_shortens_{false};
};

} // namespace trtmc::paligemma
