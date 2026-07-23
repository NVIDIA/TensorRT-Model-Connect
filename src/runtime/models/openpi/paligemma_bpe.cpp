/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/paligemma_bpe.h"

#include "runtime/models/openpi/openpi_data_plane.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace trtmc::openpi {
namespace {

constexpr std::string_view kAssetMagic{"TRTMCBPE"};
constexpr uint32_t kFlagDummyPrefix = 1U << 0U;
constexpr uint32_t kFlagRemoveExtraWhitespaces = 1U << 1U;
constexpr uint32_t kFlagEscapeWhitespaces = 1U << 2U;
constexpr uint32_t kFlagByteFallback = 1U << 3U;
constexpr uint32_t kFlagWhitespaceAsSuffix = 1U << 4U;
constexpr uint32_t kKnownFlags = kFlagDummyPrefix | kFlagRemoveExtraWhitespaces |
                                 kFlagEscapeWhitespaces | kFlagByteFallback |
                                 kFlagWhitespaceAsSuffix;
constexpr std::size_t kMaximumRecords = 1'000'000;
constexpr std::string_view kSpaceSymbol{"\xE2\x96\x81"};
constexpr std::string_view kReplacementCharacter{"\xEF\xBF\xBD"};

class AssetReader {
  public:
    explicit AssetReader(std::string_view bytes) : bytes_(bytes) {}

    uint8_t read_u8() {
        require(1);
        return static_cast<uint8_t>(bytes_[offset_++]);
    }

    uint32_t read_u32() {
        require(4);
        uint32_t value = 0;
        for (uint32_t i = 0; i < 4; ++i) {
            value |= static_cast<uint32_t>(static_cast<uint8_t>(bytes_[offset_ + i])) << (8U * i);
        }
        offset_ += 4;
        return value;
    }

    int32_t read_i32() { return static_cast<int32_t>(read_u32()); }

    float read_float() {
        const uint32_t bits = read_u32();
        float value = 0.0F;
        static_assert(sizeof(value) == sizeof(bits));
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    std::string read_string() {
        const auto size = static_cast<std::size_t>(read_u32());
        require(size);
        std::string value(bytes_.substr(offset_, size));
        offset_ += size;
        return value;
    }

    std::string read_raw(std::size_t size) {
        require(size);
        std::string value(bytes_.substr(offset_, size));
        offset_ += size;
        return value;
    }

    bool at_end() const noexcept { return offset_ == bytes_.size(); }

  private:
    void require(std::size_t size) const {
        if (size > bytes_.size() - std::min(offset_, bytes_.size())) {
            throw std::invalid_argument("Truncated OpenPI PaliGemma BPE asset");
        }
    }

    std::string_view bytes_;
    std::size_t offset_{0};
};

void append_u8(std::vector<uint8_t>& output, uint8_t value) {
    output.push_back(value);
}

void append_u32(std::vector<uint8_t>& output, uint32_t value) {
    for (uint32_t i = 0; i < 4; ++i) {
        output.push_back(static_cast<uint8_t>((value >> (8U * i)) & 0xFFU));
    }
}

void append_i32(std::vector<uint8_t>& output, int32_t value) {
    append_u32(output, static_cast<uint32_t>(value));
}

void append_float(std::vector<uint8_t>& output, float value) {
    uint32_t bits = 0;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&bits, &value, sizeof(value));
    append_u32(output, bits);
}

void append_string(std::vector<uint8_t>& output, std::string_view value) {
    if (value.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("OpenPI BPE asset string is too large");
    }
    append_u32(output, static_cast<uint32_t>(value.size()));
    output.insert(output.end(), value.begin(), value.end());
}

bool is_valid_piece_type(SentencePieceType type) {
    switch (type) {
    case SentencePieceType::kNormal:
    case SentencePieceType::kUnknown:
    case SentencePieceType::kControl:
    case SentencePieceType::kUserDefined:
    case SentencePieceType::kUnused:
    case SentencePieceType::kByte:
        return true;
    }
    return false;
}

int hex_nibble(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

bool has_byte_piece_shape(std::string_view piece) {
    return piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
           piece[5] == '>';
}

int byte_from_piece(std::string_view piece) {
    if (!has_byte_piece_shape(piece)) {
        return -1;
    }
    const int high = hex_nibble(piece[3]);
    const int low = hex_nibble(piece[4]);
    return high < 0 || low < 0 ? -1 : high * 16 + low;
}

void validate_special_id(const PaligemmaBpeAsset& asset, int32_t id, SentencePieceType type,
                         std::string_view name, bool required) {
    if (id < 0) {
        if (required) {
            throw std::invalid_argument("OpenPI BPE asset is missing " + std::string(name));
        }
        return;
    }
    if (static_cast<std::size_t>(id) >= asset.pieces.size() || asset.pieces[id].type != type) {
        throw std::invalid_argument("OpenPI BPE asset has an invalid " + std::string(name));
    }
}

void validate_piece(const BpePiece& piece, std::unordered_set<std::string>& piece_names,
                    std::array<bool, 256>& byte_found, int32_t& unknown_count) {
    if (piece.text.empty() || !std::isfinite(piece.score) || !is_valid_piece_type(piece.type)) {
        throw std::invalid_argument("OpenPI BPE asset contains an invalid piece");
    }
    if (!piece_names.insert(piece.text).second) {
        throw std::invalid_argument("OpenPI BPE asset contains a duplicate piece");
    }
    if (piece.type == SentencePieceType::kUnknown) {
        ++unknown_count;
    }
    if (piece.type != SentencePieceType::kByte) {
        return;
    }
    const int value = byte_from_piece(piece.text);
    if (value < 0 || byte_found[static_cast<std::size_t>(value)]) {
        throw std::invalid_argument("OpenPI BPE asset contains an invalid byte piece");
    }
    byte_found[static_cast<std::size_t>(value)] = true;
}

void validate_normalization_rules(const std::vector<NormalizationRule>& rules) {
    std::unordered_set<std::string> rule_sources;
    for (const auto& rule : rules) {
        if (rule.source.empty() || !rule_sources.insert(rule.source).second) {
            throw std::invalid_argument("OpenPI BPE asset contains an invalid normalization rule");
        }
    }
}

void validate_asset(const PaligemmaBpeAsset& asset) {
    if (asset.version != PaligemmaBpeAsset::kCurrentVersion) {
        throw std::invalid_argument("Unsupported OpenPI PaliGemma BPE asset version");
    }
    if (asset.pieces.empty() || asset.pieces.size() > kMaximumRecords ||
        asset.normalization_rules.size() > kMaximumRecords) {
        throw std::invalid_argument("OpenPI BPE asset record count is invalid");
    }

    std::unordered_set<std::string> piece_names;
    std::array<bool, 256> byte_found{};
    int32_t unknown_count = 0;
    for (const auto& piece : asset.pieces) {
        validate_piece(piece, piece_names, byte_found, unknown_count);
    }
    if (unknown_count != 1) {
        throw std::invalid_argument("OpenPI BPE asset must contain exactly one unknown piece");
    }
    validate_special_id(asset, asset.unknown_id, SentencePieceType::kUnknown, "unknown id", true);
    validate_special_id(asset, asset.bos_id, SentencePieceType::kControl, "BOS id", true);
    validate_special_id(asset, asset.eos_id, SentencePieceType::kControl, "EOS id", false);
    validate_special_id(asset, asset.pad_id, SentencePieceType::kControl, "padding id", false);
    if (asset.byte_fallback &&
        std::find(byte_found.begin(), byte_found.end(), false) != byte_found.end()) {
        throw std::invalid_argument(
            "OpenPI BPE byte fallback requires exactly one piece for every byte");
    }
    validate_normalization_rules(asset.normalization_rules);
}

struct Utf8Prefix {
    std::size_t length{0};
    char32_t value{0};
    char32_t minimum{0};
};

Utf8Prefix decode_utf8_prefix(uint8_t first) {
    if ((first & 0xE0U) == 0xC0U) {
        return {2, static_cast<char32_t>(first & 0x1FU), 0x80};
    }
    if ((first & 0xF0U) == 0xE0U) {
        return {3, static_cast<char32_t>(first & 0x0FU), 0x800};
    }
    if ((first & 0xF8U) == 0xF0U) {
        return {4, static_cast<char32_t>(first & 0x07U), 0x10000};
    }
    return {};
}

bool valid_unicode_scalar(char32_t value, char32_t minimum) {
    return value >= minimum && value <= 0x10FFFFU && !(value >= 0xD800U && value <= 0xDFFFU);
}

std::size_t valid_utf8_character_length(std::string_view text) {
    if (text.empty()) {
        return 0;
    }
    const auto first = static_cast<uint8_t>(text[0]);
    if (first <= 0x7FU) {
        return 1;
    }
    Utf8Prefix prefix = decode_utf8_prefix(first);
    if (prefix.length == 0 || prefix.length > text.size()) {
        return 0;
    }
    for (std::size_t i = 1; i < prefix.length; ++i) {
        const auto byte = static_cast<uint8_t>(text[i]);
        if ((byte & 0xC0U) != 0x80U) {
            return 0;
        }
        prefix.value = static_cast<char32_t>((prefix.value << 6U) | (byte & 0x3FU));
    }
    return valid_unicode_scalar(prefix.value, prefix.minimum) ? prefix.length : 0;
}

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

template <typename PrefixLookup>
std::size_t skip_leading_normalized_spaces(std::string_view text, bool remove_extra_whitespaces,
                                           PrefixLookup&& lookup) {
    if (!remove_extra_whitespaces) {
        return 0;
    }
    std::size_t consumed = 0;
    while (consumed < text.size()) {
        const auto prefix = lookup(text.substr(consumed));
        if (prefix.first != " ") {
            break;
        }
        consumed += prefix.second;
    }
    return consumed;
}

void append_normalized_whitespace(std::string& normalized, bool escape_whitespaces) {
    normalized.append(escape_whitespaces ? kSpaceSymbol : std::string_view(" "));
}

void append_normalized_piece(std::string& normalized, std::string piece, bool escape_whitespaces,
                             bool remove_extra_whitespaces, bool& previous_was_space) {
    if (previous_was_space) {
        while (!piece.empty() && piece.front() == ' ') {
            piece.erase(piece.begin());
        }
    }
    if (piece.empty()) {
        return;
    }
    for (char value : piece) {
        if (escape_whitespaces && value == ' ') {
            normalized.append(kSpaceSymbol);
        } else {
            normalized.push_back(value);
        }
    }
    previous_was_space = remove_extra_whitespaces && piece.back() == ' ';
}

template <typename PrefixLookup>
void append_normalized_text(std::string_view text, std::size_t consumed, PrefixLookup&& lookup,
                            const PaligemmaBpeAsset& asset, std::string& normalized) {
    bool previous_was_space = asset.remove_extra_whitespaces;
    while (consumed < text.size()) {
        auto prefix = lookup(text.substr(consumed));
        append_normalized_piece(normalized, std::move(prefix.first), asset.escape_whitespaces,
                                asset.remove_extra_whitespaces, previous_was_space);
        consumed += prefix.second;
    }
}

void trim_trailing_normalized_spaces(std::string& normalized, bool remove_extra_whitespaces,
                                     bool escape_whitespaces) {
    if (!remove_extra_whitespaces) {
        return;
    }
    const std::string_view space = escape_whitespaces ? kSpaceSymbol : std::string_view(" ");
    while (ends_with(normalized, space)) {
        normalized.resize(normalized.size() - space.size());
    }
}

struct EncodingSymbol {
    int32_t previous{-1};
    int32_t next{-1};
    std::size_t begin{0};
    std::size_t size{0};
    bool frozen{false};
    bool alive{true};
};

struct EncodingPiece {
    bool found{false};
    int32_t id{-1};
    float score{0.0F};
    SentencePieceType type{SentencePieceType::kNormal};
};

struct MergeCandidate {
    int32_t left{-1};
    int32_t right{-1};
    float score{0.0F};
    std::size_t size{0};
};

struct MergeCandidatePriority {
    bool operator()(const MergeCandidate& lhs, const MergeCandidate& rhs) const {
        return lhs.score < rhs.score || (lhs.score == rhs.score && lhs.left > rhs.left);
    }
};

using MergeQueue =
    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, MergeCandidatePriority>;
using ReverseUnusedMerges = std::unordered_map<std::string, std::pair<std::string, std::string>>;

std::string_view encoding_symbol_text(std::string_view text,
                                      const std::vector<EncodingSymbol>& symbols, int32_t index) {
    const auto& symbol = symbols[static_cast<std::size_t>(index)];
    return std::string_view(text.data() + symbol.begin, symbol.size);
}

template <typename UserDefinedPrefixLength>
std::pair<std::size_t, bool>
initial_symbol_length(std::string_view remaining,
                      UserDefinedPrefixLength&& user_defined_prefix_length) {
    const std::size_t user_defined_length = user_defined_prefix_length(remaining);
    if (user_defined_length != 0) {
        return {user_defined_length, true};
    }
    const std::size_t character_length = valid_utf8_character_length(remaining);
    return {character_length == 0 ? 1 : character_length, false};
}

template <typename UserDefinedPrefixLength>
std::vector<EncodingSymbol>
make_initial_symbols(std::string_view text, UserDefinedPrefixLength&& user_defined_prefix_length) {
    std::vector<EncodingSymbol> symbols;
    for (std::size_t offset = 0; offset < text.size();) {
        const std::string_view remaining(text.data() + offset, text.size() - offset);
        const auto [length, frozen] = initial_symbol_length(remaining, user_defined_prefix_length);
        const int32_t index = static_cast<int32_t>(symbols.size());
        symbols.push_back({index == 0 ? -1 : index - 1, -1, offset, length, frozen, true});
        if (index > 0) {
            symbols[static_cast<std::size_t>(index - 1)].next = index;
        }
        offset += length;
    }
    return symbols;
}

bool merge_pair_is_current(const std::vector<EncodingSymbol>& symbols, int32_t left,
                           int32_t right) {
    if (left < 0 || right < 0) {
        return false;
    }
    const auto& left_symbol = symbols[static_cast<std::size_t>(left)];
    const auto& right_symbol = symbols[static_cast<std::size_t>(right)];
    return left_symbol.alive && right_symbol.alive && !left_symbol.frozen && !right_symbol.frozen &&
           left_symbol.next == right && left_symbol.begin + left_symbol.size == right_symbol.begin;
}

template <typename PieceLookup>
void add_merge_candidate(std::string_view text, const std::vector<EncodingSymbol>& symbols,
                         int32_t left, int32_t right, PieceLookup&& lookup_piece,
                         MergeQueue& candidates, ReverseUnusedMerges& reverse_unused_merges) {
    if (!merge_pair_is_current(symbols, left, right)) {
        return;
    }
    const auto& left_symbol = symbols[static_cast<std::size_t>(left)];
    const auto& right_symbol = symbols[static_cast<std::size_t>(right)];
    const std::string merged(text.data() + left_symbol.begin, left_symbol.size + right_symbol.size);
    const EncodingPiece piece = lookup_piece(merged);
    if (!piece.found) {
        return;
    }
    candidates.push({left, right, piece.score, merged.size()});
    if (piece.type != SentencePieceType::kUnused) {
        return;
    }
    reverse_unused_merges[merged] = {std::string(encoding_symbol_text(text, symbols, left)),
                                     std::string(encoding_symbol_text(text, symbols, right))};
}

bool merge_candidate_is_current(const std::vector<EncodingSymbol>& symbols,
                                const MergeCandidate& candidate) {
    const auto& left = symbols[static_cast<std::size_t>(candidate.left)];
    const auto& right = symbols[static_cast<std::size_t>(candidate.right)];
    return left.alive && right.alive && left.next == candidate.right &&
           left.size + right.size == candidate.size;
}

void apply_merge_candidate(std::vector<EncodingSymbol>& symbols, const MergeCandidate& candidate) {
    auto& left = symbols[static_cast<std::size_t>(candidate.left)];
    auto& right = symbols[static_cast<std::size_t>(candidate.right)];
    left.size += right.size;
    left.next = right.next;
    if (right.next >= 0) {
        symbols[static_cast<std::size_t>(right.next)].previous = candidate.left;
    }
    right.alive = false;
    right.size = 0;
}

template <typename PieceLookup>
ReverseUnusedMerges merge_encoding_symbols(std::string_view text,
                                           std::vector<EncodingSymbol>& symbols,
                                           PieceLookup&& lookup_piece) {
    MergeQueue candidates;
    ReverseUnusedMerges reverse_unused_merges;
    for (std::size_t i = 1; i < symbols.size(); ++i) {
        add_merge_candidate(text, symbols, static_cast<int32_t>(i - 1), static_cast<int32_t>(i),
                            lookup_piece, candidates, reverse_unused_merges);
    }
    while (!candidates.empty()) {
        const MergeCandidate candidate = candidates.top();
        candidates.pop();
        if (!merge_candidate_is_current(symbols, candidate)) {
            continue;
        }
        apply_merge_candidate(symbols, candidate);
        const auto& merged = symbols[static_cast<std::size_t>(candidate.left)];
        add_merge_candidate(text, symbols, merged.previous, candidate.left, lookup_piece,
                            candidates, reverse_unused_merges);
        add_merge_candidate(text, symbols, candidate.left, merged.next, lookup_piece, candidates,
                            reverse_unused_merges);
    }
    return reverse_unused_merges;
}

void emit_byte_fallback(std::string_view piece, const std::array<int32_t, 256>& byte_ids,
                        std::vector<int32_t>& ids) {
    for (const unsigned char byte : piece) {
        const int32_t byte_id = byte_ids[byte];
        if (byte_id < 0) {
            throw std::runtime_error("OpenPI BPE byte fallback table is incomplete");
        }
        ids.push_back(byte_id);
    }
}

template <typename PieceLookup>
void emit_piece(std::string_view piece, const PaligemmaBpeAsset& asset,
                const std::array<int32_t, 256>& byte_ids,
                const ReverseUnusedMerges& reverse_unused_merges, PieceLookup&& lookup_piece,
                std::vector<int32_t>& ids) {
    const EncodingPiece found = lookup_piece(piece);
    const int32_t id = found.found ? found.id : asset.unknown_id;
    const auto type = asset.pieces[static_cast<std::size_t>(id)].type;
    if (type == SentencePieceType::kUnused) {
        const auto reverse = reverse_unused_merges.find(std::string(piece));
        if (reverse != reverse_unused_merges.end()) {
            emit_piece(reverse->second.first, asset, byte_ids, reverse_unused_merges, lookup_piece,
                       ids);
            emit_piece(reverse->second.second, asset, byte_ids, reverse_unused_merges, lookup_piece,
                       ids);
            return;
        }
    }
    if (type == SentencePieceType::kUnknown && asset.byte_fallback) {
        emit_byte_fallback(piece, byte_ids, ids);
        return;
    }
    ids.push_back(id);
}

template <typename PieceLookup>
std::vector<int32_t>
emit_encoding(std::string_view text, const std::vector<EncodingSymbol>& symbols,
              const PaligemmaBpeAsset& asset, const std::array<int32_t, 256>& byte_ids,
              const ReverseUnusedMerges& reverse_unused_merges, PieceLookup&& lookup_piece,
              bool add_bos) {
    std::vector<int32_t> ids;
    if (add_bos) {
        ids.push_back(asset.bos_id);
    }
    for (int32_t index = symbols.empty() ? -1 : 0; index >= 0;
         index = symbols[static_cast<std::size_t>(index)].next) {
        emit_piece(encoding_symbol_text(text, symbols, index), asset, byte_ids,
                   reverse_unused_merges, lookup_piece, ids);
    }
    return ids;
}

} // namespace

PaligemmaBpeAsset parse_paligemma_bpe_asset(std::string_view bytes) {
    AssetReader reader(bytes);
    if (reader.read_raw(kAssetMagic.size()) != kAssetMagic) {
        throw std::invalid_argument("OpenPI PaliGemma BPE asset has invalid magic");
    }

    PaligemmaBpeAsset asset;
    asset.version = reader.read_u32();
    const uint32_t flags = reader.read_u32();
    if ((flags & ~kKnownFlags) != 0U) {
        throw std::invalid_argument("OpenPI PaliGemma BPE asset has unknown flags");
    }
    asset.add_dummy_prefix = (flags & kFlagDummyPrefix) != 0U;
    asset.remove_extra_whitespaces = (flags & kFlagRemoveExtraWhitespaces) != 0U;
    asset.escape_whitespaces = (flags & kFlagEscapeWhitespaces) != 0U;
    asset.byte_fallback = (flags & kFlagByteFallback) != 0U;
    asset.treat_whitespace_as_suffix = (flags & kFlagWhitespaceAsSuffix) != 0U;
    asset.unknown_id = reader.read_i32();
    asset.bos_id = reader.read_i32();
    asset.eos_id = reader.read_i32();
    asset.pad_id = reader.read_i32();
    const auto piece_count = static_cast<std::size_t>(reader.read_u32());
    const auto rule_count = static_cast<std::size_t>(reader.read_u32());
    if (piece_count > kMaximumRecords || rule_count > kMaximumRecords) {
        throw std::invalid_argument("OpenPI BPE asset record count exceeds its limit");
    }

    asset.pieces.reserve(piece_count);
    for (std::size_t i = 0; i < piece_count; ++i) {
        BpePiece piece;
        piece.type = static_cast<SentencePieceType>(reader.read_u8());
        piece.score = reader.read_float();
        piece.text = reader.read_string();
        asset.pieces.push_back(std::move(piece));
    }
    asset.normalization_rules.reserve(rule_count);
    for (std::size_t i = 0; i < rule_count; ++i) {
        NormalizationRule rule;
        rule.source = reader.read_string();
        rule.replacement = reader.read_string();
        asset.normalization_rules.push_back(std::move(rule));
    }
    if (!reader.at_end()) {
        throw std::invalid_argument("OpenPI PaliGemma BPE asset contains trailing bytes");
    }
    validate_asset(asset);
    return asset;
}

std::vector<uint8_t> serialize_paligemma_bpe_asset(const PaligemmaBpeAsset& asset) {
    validate_asset(asset);
    if (asset.pieces.size() > std::numeric_limits<uint32_t>::max() ||
        asset.normalization_rules.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("OpenPI BPE asset has too many records");
    }
    std::vector<uint8_t> output;
    output.insert(output.end(), kAssetMagic.begin(), kAssetMagic.end());
    append_u32(output, asset.version);
    uint32_t flags = 0;
    flags |= asset.add_dummy_prefix ? kFlagDummyPrefix : 0U;
    flags |= asset.remove_extra_whitespaces ? kFlagRemoveExtraWhitespaces : 0U;
    flags |= asset.escape_whitespaces ? kFlagEscapeWhitespaces : 0U;
    flags |= asset.byte_fallback ? kFlagByteFallback : 0U;
    flags |= asset.treat_whitespace_as_suffix ? kFlagWhitespaceAsSuffix : 0U;
    append_u32(output, flags);
    append_i32(output, asset.unknown_id);
    append_i32(output, asset.bos_id);
    append_i32(output, asset.eos_id);
    append_i32(output, asset.pad_id);
    append_u32(output, static_cast<uint32_t>(asset.pieces.size()));
    append_u32(output, static_cast<uint32_t>(asset.normalization_rules.size()));
    for (const auto& piece : asset.pieces) {
        append_u8(output, static_cast<uint8_t>(piece.type));
        append_float(output, piece.score);
        append_string(output, piece.text);
    }
    for (const auto& rule : asset.normalization_rules) {
        append_string(output, rule.source);
        append_string(output, rule.replacement);
    }
    return output;
}

PaligemmaBpeTokenizer::PaligemmaBpeTokenizer(PaligemmaBpeAsset asset) : asset_(std::move(asset)) {
    validate_asset(asset_);
    byte_ids_.fill(-1);
    for (std::size_t i = 0; i < asset_.pieces.size(); ++i) {
        const auto& piece = asset_.pieces[i];
        if (piece.type == SentencePieceType::kNormal ||
            piece.type == SentencePieceType::kUserDefined ||
            piece.type == SentencePieceType::kUnused) {
            merge_pieces_.emplace(piece.text,
                                  PieceIndex{static_cast<int32_t>(i), piece.score, piece.type});
        }
        if (piece.type == SentencePieceType::kUserDefined) {
            const auto first = static_cast<unsigned char>(piece.text.front());
            user_defined_pieces_by_first_byte_[first].push_back(piece.text);
        }
        if (piece.type == SentencePieceType::kByte) {
            byte_ids_[static_cast<std::size_t>(byte_from_piece(piece.text))] =
                static_cast<int32_t>(i);
        }
    }
    for (auto& pieces : user_defined_pieces_by_first_byte_) {
        std::sort(pieces.begin(), pieces.end(), [](const std::string& lhs, const std::string& rhs) {
            return lhs.size() > rhs.size();
        });
    }
    normalization_rules_ = asset_.normalization_rules;
    std::stable_sort(normalization_rules_.begin(), normalization_rules_.end(),
                     [](const NormalizationRule& lhs, const NormalizationRule& rhs) {
                         return lhs.source.size() > rhs.source.size();
                     });
}

std::size_t PaligemmaBpeTokenizer::user_defined_prefix_length(std::string_view text) const {
    if (text.empty()) {
        return 0;
    }
    const auto first = static_cast<unsigned char>(text.front());
    for (const auto& piece : user_defined_pieces_by_first_byte_[first]) {
        if (starts_with(text, piece)) {
            return piece.size();
        }
    }
    return 0;
}

std::pair<std::string, std::size_t>
PaligemmaBpeTokenizer::normalized_prefix(std::string_view text) const {
    if (text.empty()) {
        return {};
    }
    const std::size_t user_defined_length = user_defined_prefix_length(text);
    if (user_defined_length != 0) {
        return {std::string(text.substr(0, user_defined_length)), user_defined_length};
    }
    for (const auto& rule : normalization_rules_) {
        if (starts_with(text, rule.source)) {
            return {rule.replacement, rule.source.size()};
        }
    }
    const std::size_t character_length = valid_utf8_character_length(text);
    if (character_length == 0) {
        return {std::string(kReplacementCharacter), 1};
    }
    return {std::string(text.substr(0, character_length)), character_length};
}

std::string PaligemmaBpeTokenizer::normalize(std::string_view text) const {
    if (text.empty()) {
        return {};
    }

    const auto lookup_prefix = [this](std::string_view remaining) {
        return normalized_prefix(remaining);
    };
    const std::size_t consumed =
        skip_leading_normalized_spaces(text, asset_.remove_extra_whitespaces, lookup_prefix);
    if (consumed == text.size()) {
        return {};
    }

    std::string normalized;
    if (!asset_.treat_whitespace_as_suffix && asset_.add_dummy_prefix) {
        append_normalized_whitespace(normalized, asset_.escape_whitespaces);
    }

    append_normalized_text(text, consumed, lookup_prefix, asset_, normalized);
    trim_trailing_normalized_spaces(normalized, asset_.remove_extra_whitespaces,
                                    asset_.escape_whitespaces);
    if (asset_.treat_whitespace_as_suffix && asset_.add_dummy_prefix) {
        append_normalized_whitespace(normalized, asset_.escape_whitespaces);
    }
    return normalized;
}

std::vector<int32_t> PaligemmaBpeTokenizer::encode(std::string_view text, bool add_bos) const {
    const std::string normalized_text = normalize(text);
    auto symbols = make_initial_symbols(normalized_text, [this](std::string_view remaining) {
        return user_defined_prefix_length(remaining);
    });
    const auto lookup_piece = [this](std::string_view piece) {
        const auto found = merge_pieces_.find(std::string(piece));
        if (found == merge_pieces_.end()) {
            return EncodingPiece{};
        }
        return EncodingPiece{true, found->second.id, found->second.score, found->second.type};
    };
    const ReverseUnusedMerges reverse_unused_merges =
        merge_encoding_symbols(normalized_text, symbols, lookup_piece);
    return emit_encoding(normalized_text, symbols, asset_, byte_ids_, reverse_unused_merges,
                         lookup_piece, add_bos);
}

TokenizedPrompt PaligemmaBpeTokenizer::pad_or_truncate(std::vector<int32_t> ids,
                                                       std::size_t max_length) const {
    if (max_length == 0) {
        throw std::invalid_argument("OpenPI prompt maximum length must be positive");
    }
    TokenizedPrompt result;
    result.truncated = ids.size() > max_length;
    if (ids.size() > max_length) {
        ids.resize(max_length);
    }
    result.token_mask.assign(ids.size(), 1U);
    ids.resize(max_length, 0);
    result.token_mask.resize(max_length, 0U);
    result.token_ids = std::move(ids);
    return result;
}

TokenizedPrompt PaligemmaBpeTokenizer::tokenize_pi05(std::string_view prompt,
                                                     const std::vector<float>& state,
                                                     std::size_t max_length) const {
    return pad_or_truncate(encode(format_pi05_prompt(prompt, state), true), max_length);
}

TokenizedPrompt PaligemmaBpeTokenizer::tokenize_pi0(std::string_view prompt,
                                                    std::size_t max_length) const {
    auto ids = encode(clean_prompt(prompt), true);
    auto newline_ids = encode("\n", false);
    ids.insert(ids.end(), newline_ids.begin(), newline_ids.end());
    return pad_or_truncate(std::move(ids), max_length);
}

} // namespace trtmc::openpi
