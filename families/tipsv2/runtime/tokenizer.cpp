/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/tipsv2/runtime/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace trtmc::tipsv2 {
namespace {

constexpr std::string_view kAssetMagic = "TRTMCTIP";
constexpr std::string_view kSpaceSymbol = "\xE2\x96\x81";          // U+2581 LOWER ONE EIGHTH BLOCK
constexpr std::string_view kReplacementCharacter = "\xEF\xBF\xBD"; // U+FFFD
// The one character in Unicode whose lower case depends on its neighbours.
constexpr std::uint32_t kCapitalSigma = 0x03A3;
constexpr std::string_view kSmallSigma = "\xCF\x83"; // U+03C3
constexpr std::string_view kFinalSigma = "\xCF\x82"; // U+03C2
// A corrupt length must not become a huge allocation; no SentencePiece model this family accepts
// is anywhere near this.
constexpr std::size_t kMaximumRecords = 1000000;

// The ranges are sorted and disjoint, so one binary search answers membership.
bool in_ranges(const std::vector<CodePointRange>& ranges, std::uint32_t code_point) {
    const auto found = std::upper_bound(
        ranges.begin(), ranges.end(), code_point,
        [](std::uint32_t value, const CodePointRange& range) { return value < range.first; });
    return found != ranges.begin() && code_point <= std::prev(found)->last;
}

constexpr std::uint32_t kFlagAddDummyPrefix = 1U << 0U;
constexpr std::uint32_t kFlagRemoveExtraWhitespaces = 1U << 1U;
constexpr std::uint32_t kFlagEscapeWhitespaces = 1U << 2U;
constexpr std::uint32_t kFlagByteFallback = 1U << 3U;
constexpr std::uint32_t kFlagTreatWhitespaceAsSuffix = 1U << 4U;
constexpr std::uint32_t kAllFlags = kFlagAddDummyPrefix | kFlagRemoveExtraWhitespaces |
                                    kFlagEscapeWhitespaces | kFlagByteFallback |
                                    kFlagTreatWhitespaceAsSuffix;

class AssetReader {
  public:
    explicit AssetReader(std::string_view bytes) : bytes_(bytes) {}

    std::string_view take(std::size_t length) {
        if (length > bytes_.size() - offset_) {
            throw std::invalid_argument("tipsv2 tokenizer asset is truncated");
        }
        const auto chunk = bytes_.substr(offset_, length);
        offset_ += length;
        return chunk;
    }

    std::uint32_t u32() {
        const auto raw = take(4);
        return static_cast<std::uint32_t>(static_cast<unsigned char>(raw[0])) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(raw[1])) << 8U) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(raw[2])) << 16U) |
               (static_cast<std::uint32_t>(static_cast<unsigned char>(raw[3])) << 24U);
    }

    std::int32_t i32() { return static_cast<std::int32_t>(u32()); }

    float f32() {
        const std::uint32_t raw = u32();
        float value = 0.0F;
        static_assert(sizeof(value) == sizeof(raw), "float is not 32 bits");
        std::memcpy(&value, &raw, sizeof(value));
        return value;
    }

    std::uint8_t u8() { return static_cast<std::uint8_t>(take(1)[0]); }

    void finish() const {
        if (offset_ != bytes_.size()) {
            throw std::invalid_argument("tipsv2 tokenizer asset has trailing bytes");
        }
    }

  private:
    std::string_view bytes_;
    std::size_t offset_{0};
};

bool is_piece_type(std::uint8_t value) {
    return value >= static_cast<std::uint8_t>(PieceType::kNormal) &&
           value <= static_cast<std::uint8_t>(PieceType::kByte);
}

int byte_from_piece(std::string_view piece) {
    // The build step guarantees the spelling; this parses it back without trusting the index.
    if (piece.size() != 6 || piece.substr(0, 3) != "<0x" || piece[5] != '>') {
        return -1;
    }
    int value = 0;
    for (std::size_t index = 3; index < 5; ++index) {
        const char digit = piece[index];
        int nibble = 0;
        if (digit >= '0' && digit <= '9') {
            nibble = digit - '0';
        } else if (digit >= 'A' && digit <= 'F') {
            nibble = digit - 'A' + 10;
        } else {
            return -1;
        }
        value = value * 16 + nibble;
    }
    return value;
}

void validate_asset(const TokenizerAsset& asset) {
    if (asset.version != TokenizerAsset::kCurrentVersion) {
        throw std::invalid_argument("tipsv2 tokenizer asset version is not supported");
    }
    if (asset.pieces.empty() || asset.pieces.size() > kMaximumRecords ||
        asset.lowercase.size() > kMaximumRecords) {
        throw std::invalid_argument("tipsv2 tokenizer asset record count is invalid");
    }
    if (asset.charsmap.empty()) {
        throw std::invalid_argument("tipsv2 tokenizer asset carries no normalization map");
    }

    std::unordered_set<std::string> seen;
    std::array<bool, 256> byte_found{};
    int unknown_count = 0;
    for (std::size_t index = 0; index < asset.pieces.size(); ++index) {
        const auto& piece = asset.pieces[index];
        if (piece.text.empty()) {
            // An empty piece would match at every position without advancing.
            throw std::invalid_argument("tipsv2 tokenizer piece is empty");
        }
        if (!seen.insert(piece.text).second) {
            throw std::invalid_argument("tipsv2 tokenizer has duplicate pieces");
        }
        if (!(piece.score == piece.score) ||
            piece.score == std::numeric_limits<float>::infinity() ||
            piece.score == -std::numeric_limits<float>::infinity()) {
            throw std::invalid_argument("tipsv2 tokenizer piece has a non-finite score");
        }
        if (piece.type == PieceType::kUnknown) {
            ++unknown_count;
        }
        if (piece.type == PieceType::kByte) {
            const int value = byte_from_piece(piece.text);
            if (value < 0 || byte_found[static_cast<std::size_t>(value)]) {
                throw std::invalid_argument("tipsv2 tokenizer has an invalid byte piece");
            }
            byte_found[static_cast<std::size_t>(value)] = true;
        }
        static_cast<void>(index);
    }
    if (unknown_count != 1) {
        throw std::invalid_argument("tipsv2 tokenizer must have exactly one unknown piece");
    }

    const auto count = static_cast<std::int32_t>(asset.pieces.size());
    const auto check = [count, &asset](std::int32_t id, PieceType expected, bool required,
                                       const char* what) {
        if (id < 0) {
            if (required) {
                throw std::invalid_argument(std::string("tipsv2 tokenizer has no ") + what);
            }
            return;
        }
        if (id >= count) {
            throw std::invalid_argument(std::string("tipsv2 tokenizer ") + what +
                                        " is outside the vocabulary");
        }
        if (asset.pieces[static_cast<std::size_t>(id)].type != expected) {
            throw std::invalid_argument(std::string("tipsv2 tokenizer ") + what +
                                        " has the wrong piece type");
        }
    };
    check(asset.unknown_id, PieceType::kUnknown, true, "unknown id");
    check(asset.bos_id, PieceType::kControl, false, "BOS id");
    check(asset.eos_id, PieceType::kControl, false, "EOS id");
    // The text engine derives its padding mask from this id, so it has to be a piece no prompt
    // can produce - which is what a control piece is.
    check(asset.pad_id, PieceType::kControl, true, "padding id");

    std::vector<std::int32_t> enabled;
    for (const std::int32_t id : {asset.unknown_id, asset.bos_id, asset.eos_id, asset.pad_id}) {
        if (id >= 0) {
            enabled.push_back(id);
        }
    }
    std::sort(enabled.begin(), enabled.end());
    if (std::adjacent_find(enabled.begin(), enabled.end()) != enabled.end()) {
        throw std::invalid_argument("tipsv2 tokenizer special ids are not distinct");
    }

    const bool all_bytes =
        std::find(byte_found.begin(), byte_found.end(), false) == byte_found.end();
    const bool any_byte = std::find(byte_found.begin(), byte_found.end(), true) != byte_found.end();
    if (asset.byte_fallback && !all_bytes) {
        throw std::invalid_argument(
            "tipsv2 tokenizer byte fallback needs one piece for every byte");
    }
    if (!asset.byte_fallback) {
        // Required, not optional: without it the reference answers a run of unknown characters
        // with one unknown id and this encoder emits one per symbol. A bundle built for that
        // mode is refused rather than tokenized by a rule this family was never qualified on.
        throw std::invalid_argument("tipsv2 needs a tokenizer with byte fallback");
    }
    static_cast<void>(any_byte);

    std::int64_t previous = -1;
    for (const auto& entry : asset.lowercase) {
        if (static_cast<std::int64_t>(entry.code_point) <= previous) {
            throw std::invalid_argument("tipsv2 lowercase table is not sorted by code point");
        }
        if (entry.code_point > 0x10FFFFU ||
            (entry.code_point >= 0xD800U && entry.code_point <= 0xDFFFU)) {
            throw std::invalid_argument("tipsv2 lowercase table has an invalid code point");
        }
        if (entry.lowered.empty()) {
            throw std::invalid_argument("tipsv2 lowercase table maps a code point to nothing");
        }
        previous = static_cast<std::int64_t>(entry.code_point);
    }

    // `in_ranges` binary-searches these, so they have to be sorted, non-empty and disjoint.
    for (const auto* table : {&asset.cased, &asset.case_ignorable}) {
        std::int64_t end = -2;
        for (const auto& range : *table) {
            if (range.first > range.last || range.last > 0x10FFFFU ||
                static_cast<std::int64_t>(range.first) <= end + 1) {
                throw std::invalid_argument("tipsv2 case table ranges are not sorted and disjoint");
            }
            end = static_cast<std::int64_t>(range.last);
        }
    }
    if (asset.cased.empty() || asset.case_ignorable.empty()) {
        // Either one missing gives every sigma the same form, silently: with no cased table the
        // scan after a sigma finds nothing cased and it never takes the final form.
        throw std::invalid_argument("tipsv2 needs both case tables");
    }
}

struct Utf8Prefix {
    std::size_t length{0};
    char32_t value{0};
    char32_t minimum{0};
};

Utf8Prefix decode_utf8_prefix(std::uint8_t first) {
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

// Length of the valid UTF-8 character starting at `text`, or 0. Every continuation byte must be
// present, the value must not be an overlong encoding, a surrogate, or above U+10FFFF.
std::size_t valid_utf8_character_length(std::string_view text, char32_t* code_point = nullptr) {
    if (text.empty()) {
        return 0;
    }
    const auto first = static_cast<std::uint8_t>(text[0]);
    if (first <= 0x7FU) {
        if (code_point != nullptr) {
            *code_point = static_cast<char32_t>(first);
        }
        return 1;
    }
    Utf8Prefix prefix = decode_utf8_prefix(first);
    if (prefix.length == 0 || prefix.length > text.size()) {
        return 0;
    }
    for (std::size_t index = 1; index < prefix.length; ++index) {
        const auto byte = static_cast<std::uint8_t>(text[index]);
        if ((byte & 0xC0U) != 0x80U) {
            return 0;
        }
        prefix.value = static_cast<char32_t>((prefix.value << 6U) | (byte & 0x3FU));
    }
    if (prefix.value < prefix.minimum || prefix.value > 0x10FFFFU ||
        (prefix.value >= 0xD800U && prefix.value <= 0xDFFFU)) {
        return 0;
    }
    if (code_point != nullptr) {
        *code_point = prefix.value;
    }
    return prefix.length;
}

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

struct EncodingSymbol {
    std::int32_t previous{-1};
    std::int32_t next{-1};
    std::size_t begin{0};
    std::size_t size{0};
    bool frozen{false};
    bool alive{true};
};

struct MergeCandidate {
    std::int32_t left{-1};
    std::int32_t right{-1};
    float score{0.0F};
    std::size_t size{0};
};

// SentencePiece's SymbolPairComparator: the highest score merges first, and a tie goes to the
// leftmost pair. `std::priority_queue` is a max-heap over this ordering.
struct MergeCandidatePriority {
    bool operator()(const MergeCandidate& lhs, const MergeCandidate& rhs) const {
        return lhs.score < rhs.score || (lhs.score == rhs.score && lhs.left > rhs.left);
    }
};

using MergeQueue =
    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, MergeCandidatePriority>;
using ReverseUnusedMerges = std::unordered_map<std::string, std::pair<std::string, std::string>>;

std::string_view symbol_text(std::string_view text, const std::vector<EncodingSymbol>& symbols,
                             std::int32_t index) {
    const auto& symbol = symbols[static_cast<std::size_t>(index)];
    return text.substr(symbol.begin, symbol.size);
}

} // namespace

TokenizerAsset parse_tokenizer_asset(std::string_view bytes) {
    AssetReader reader(bytes);
    if (reader.take(kAssetMagic.size()) != kAssetMagic) {
        throw std::invalid_argument("tipsv2 tokenizer asset has the wrong magic");
    }
    TokenizerAsset asset;
    asset.version = reader.u32();
    if (asset.version != TokenizerAsset::kCurrentVersion) {
        // Before anything else is read: another version may lay its records out differently, so
        // parsing on would be reading this one's layout out of those bytes.
        throw std::invalid_argument("tipsv2 tokenizer asset version is not supported");
    }
    const std::uint32_t flags = reader.u32();
    if ((flags & ~kAllFlags) != 0U) {
        throw std::invalid_argument("tipsv2 tokenizer asset has unknown flags");
    }
    asset.add_dummy_prefix = (flags & kFlagAddDummyPrefix) != 0U;
    asset.remove_extra_whitespaces = (flags & kFlagRemoveExtraWhitespaces) != 0U;
    asset.escape_whitespaces = (flags & kFlagEscapeWhitespaces) != 0U;
    asset.byte_fallback = (flags & kFlagByteFallback) != 0U;
    asset.treat_whitespace_as_suffix = (flags & kFlagTreatWhitespaceAsSuffix) != 0U;
    asset.unknown_id = reader.i32();
    asset.bos_id = reader.i32();
    asset.eos_id = reader.i32();
    asset.pad_id = reader.i32();
    const std::uint32_t piece_count = reader.u32();
    const std::uint32_t charsmap_length = reader.u32();
    const std::uint32_t lowercase_count = reader.u32();
    const std::uint32_t cased_count = reader.u32();
    const std::uint32_t case_ignorable_count = reader.u32();
    if (piece_count > kMaximumRecords || lowercase_count > kMaximumRecords ||
        cased_count > kMaximumRecords || case_ignorable_count > kMaximumRecords) {
        throw std::invalid_argument("tipsv2 tokenizer asset record count is invalid");
    }

    asset.pieces.reserve(piece_count);
    for (std::uint32_t index = 0; index < piece_count; ++index) {
        TokenizerPiece piece;
        const std::uint8_t type = reader.u8();
        if (!is_piece_type(type)) {
            throw std::invalid_argument("tipsv2 tokenizer piece has an unknown type");
        }
        piece.type = static_cast<PieceType>(type);
        piece.score = reader.f32();
        piece.text = std::string(reader.take(reader.u32()));
        asset.pieces.push_back(std::move(piece));
    }
    asset.charsmap = std::string(reader.take(charsmap_length));
    asset.lowercase.reserve(lowercase_count);
    for (std::uint32_t index = 0; index < lowercase_count; ++index) {
        LowercaseEntry entry;
        entry.code_point = reader.u32();
        entry.lowered = std::string(reader.take(reader.u32()));
        asset.lowercase.push_back(std::move(entry));
    }
    for (auto* table : {&asset.cased, &asset.case_ignorable}) {
        const std::uint32_t count = table == &asset.cased ? cased_count : case_ignorable_count;
        table->reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            CodePointRange range;
            range.first = reader.u32();
            range.last = reader.u32();
            table->push_back(range);
        }
    }
    reader.finish();
    return asset;
}

Tokenizer::Tokenizer(TokenizerAsset asset) : asset_(std::move(asset)) {
    validate_asset(asset_);
    charsmap_.load(asset_.charsmap);
    byte_ids_.fill(-1);
    for (std::size_t index = 0; index < asset_.pieces.size(); ++index) {
        const auto& piece = asset_.pieces[index];
        longest_piece_bytes_ = std::max(longest_piece_bytes_, piece.text.size());
        // Only these three may be produced by merging text. Control, unknown and byte pieces are
        // left out, which is what keeps a prompt from ever naming a special token.
        if (piece.type == PieceType::kNormal || piece.type == PieceType::kUserDefined ||
            piece.type == PieceType::kUnused) {
            merge_pieces_.emplace(
                piece.text, PieceIndex{static_cast<std::int32_t>(index), piece.score, piece.type});
        }
        if (piece.type == PieceType::kUserDefined) {
            const auto first = static_cast<unsigned char>(piece.text.front());
            user_defined_by_first_byte_[first].push_back(piece.text);
        }
        if (piece.type == PieceType::kByte) {
            // `byte_from_piece` answers -1 for a byte piece not spelled `<0xAB>`. `validate_asset`
            // has already refused such a piece, but this is a write into a 256-entry array driven
            // by a bundle section, so it is guarded where it happens rather than only upstream.
            const int value = byte_from_piece(piece.text);
            if (value >= 0) {
                byte_ids_[static_cast<std::size_t>(value)] = static_cast<std::int32_t>(index);
            }
        }
    }
    for (auto& bucket : user_defined_by_first_byte_) {
        std::sort(bucket.begin(), bucket.end(), [](const std::string& lhs, const std::string& rhs) {
            return lhs.size() > rhs.size();
        });
    }
    lowercase_.reserve(asset_.lowercase.size());
    for (const auto& entry : asset_.lowercase) {
        lowercase_.emplace(entry.code_point, entry.lowered);
    }
}

bool Tokenizer::is_cased(std::uint32_t code_point) const {
    return in_ranges(asset_.cased, code_point);
}

bool Tokenizer::is_case_ignorable(std::uint32_t code_point) const {
    return in_ranges(asset_.case_ignorable, code_point);
}

std::string Tokenizer::lowercase(std::string_view text) const {
    // Python's `str.lower()`, replayed from the tables the build step read out of the interpreter
    // that produced the bundle. A byte that is not part of a valid UTF-8 character keeps its own
    // position and value: the reference is handed a `str` and cannot see one, and the normalizer
    // turns it into U+FFFD afterwards. Such a byte is neither cased nor case-ignorable, so it
    // ends a sigma's context exactly as a punctuation mark would.
    //
    // The input is decoded first because one character's mapping depends on its neighbours:
    // U+03A3 lowercases to the final sigma U+03C2 when a cased character precedes it and none
    // follows, skipping case-ignorable characters in both directions. CPython's
    // `handle_capital_sigma` is the shape below, and it is the only such character in Unicode.
    struct Decoded {
        std::uint32_t code_point;
        std::size_t begin;
        std::size_t size;
        bool valid;
    };
    std::vector<Decoded> characters;
    characters.reserve(text.size());
    for (std::size_t offset = 0; offset < text.size();) {
        char32_t code_point = 0;
        const std::size_t length = valid_utf8_character_length(text.substr(offset), &code_point);
        if (length == 0) {
            characters.push_back({0, offset, 1, false});
            offset += 1;
            continue;
        }
        characters.push_back({static_cast<std::uint32_t>(code_point), offset, length, true});
        offset += length;
    }

    const auto cased_neighbour = [this, &characters](std::size_t index, bool backwards) {
        // Walk away from `index`, skipping case-ignorable characters, and report whether the
        // first character that stops the walk is cased. Running off the end is "not cased".
        std::size_t position = index;
        while (true) {
            if (backwards) {
                if (position == 0) {
                    return false;
                }
                --position;
            } else {
                ++position;
                if (position >= characters.size()) {
                    return false;
                }
            }
            const auto& neighbour = characters[position];
            if (!neighbour.valid || !is_case_ignorable(neighbour.code_point)) {
                return neighbour.valid && is_cased(neighbour.code_point);
            }
        }
    };

    std::string out;
    out.reserve(text.size());
    for (std::size_t index = 0; index < characters.size(); ++index) {
        const auto& character = characters[index];
        if (!character.valid) {
            out.append(text.substr(character.begin, character.size));
            continue;
        }
        if (character.code_point == kCapitalSigma) {
            const bool final_form = cased_neighbour(index, /*backwards=*/true) &&
                                    !cased_neighbour(index, /*backwards=*/false);
            out.append(final_form ? kFinalSigma : kSmallSigma);
            continue;
        }
        const auto found = lowercase_.find(character.code_point);
        if (found == lowercase_.end()) {
            out.append(text.substr(character.begin, character.size));
        } else {
            out.append(found->second);
        }
    }
    return out;
}

std::size_t Tokenizer::user_defined_prefix_length(std::string_view text) const {
    if (text.empty()) {
        return 0;
    }
    const auto first = static_cast<unsigned char>(text.front());
    for (const auto& piece : user_defined_by_first_byte_[first]) {
        if (starts_with(text, piece)) {
            return piece.size();
        }
    }
    return 0;
}

std::pair<std::string, std::size_t> Tokenizer::normalized_prefix(std::string_view text) const {
    if (text.empty()) {
        return {};
    }
    // A user-defined symbol passes through the normalizer untouched; that is why it is checked
    // before the map and not after.
    const std::size_t user_defined = user_defined_prefix_length(text);
    if (user_defined != 0) {
        return {std::string(text.substr(0, user_defined)), user_defined};
    }
    const auto mapped = charsmap_.prefix(text);
    if (mapped.second != 0) {
        return {std::string(mapped.first), mapped.second};
    }
    const std::size_t length = valid_utf8_character_length(text);
    if (length == 0) {
        // One U+FFFD for one bad byte, as SentencePiece's normalizer does, so a malformed input
        // still terminates instead of throwing.
        return {std::string(kReplacementCharacter), 1};
    }
    return {std::string(text.substr(0, length)), length};
}

std::string Tokenizer::normalize(std::string_view text) const {
    if (text.empty()) {
        return {};
    }
    const std::string_view space = asset_.escape_whitespaces ? kSpaceSymbol : std::string_view(" ");

    std::size_t consumed = 0;
    if (asset_.remove_extra_whitespaces) {
        while (consumed < text.size()) {
            const auto prefix = normalized_prefix(text.substr(consumed));
            if (prefix.first != " ") {
                break;
            }
            consumed += prefix.second;
        }
    }
    if (consumed == text.size()) {
        // Nothing but whitespace: the reference returns an empty normalization, not a lone dummy
        // prefix.
        return {};
    }

    std::string normalized;
    normalized.reserve(text.size() + space.size());
    if (!asset_.treat_whitespace_as_suffix && asset_.add_dummy_prefix) {
        normalized.append(space);
    }

    // Seeded true so that the collapse carries on into the first piece after the leading spaces
    // were skipped.
    bool previous_was_space = asset_.remove_extra_whitespaces;
    while (consumed < text.size()) {
        auto prefix = normalized_prefix(text.substr(consumed));
        consumed += prefix.second;
        std::string piece = std::move(prefix.first);
        if (previous_was_space) {
            std::size_t lead = 0;
            while (lead < piece.size() && piece[lead] == ' ') {
                ++lead;
            }
            piece.erase(0, lead);
        }
        if (piece.empty()) {
            continue;
        }
        for (const char value : piece) {
            if (asset_.escape_whitespaces && value == ' ') {
                normalized.append(kSpaceSymbol);
            } else {
                normalized.push_back(value);
            }
        }
        previous_was_space = asset_.remove_extra_whitespaces && piece.back() == ' ';
    }

    if (asset_.remove_extra_whitespaces) {
        while (ends_with(normalized, space)) {
            normalized.resize(normalized.size() - space.size());
        }
    }
    if (asset_.treat_whitespace_as_suffix && asset_.add_dummy_prefix) {
        normalized.append(space);
    }
    return normalized;
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text) const {
    const std::string normalized = normalize(lowercase(text));

    std::vector<EncodingSymbol> symbols;
    for (std::size_t offset = 0; offset < normalized.size();) {
        const std::string_view remaining = std::string_view(normalized).substr(offset);
        std::size_t length = user_defined_prefix_length(remaining);
        const bool frozen = length != 0;
        if (!frozen) {
            length = valid_utf8_character_length(remaining);
            if (length == 0) {
                length = 1;
            }
        }
        if (symbols.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            // The links between symbols are int32 indices. The pipeline bounds a prompt long
            // before this, but the tokenizer is usable on its own and must not wrap instead.
            throw std::invalid_argument("tipsv2 prompt has too many symbols to tokenize");
        }
        const auto index = static_cast<std::int32_t>(symbols.size());
        symbols.push_back({index == 0 ? -1 : index - 1, -1, offset, length, frozen, true});
        if (index > 0) {
            symbols[static_cast<std::size_t>(index - 1)].next = index;
        }
        offset += length;
    }

    const auto lookup = [this](std::string_view piece) -> const PieceIndex* {
        const auto found = merge_pieces_.find(std::string(piece));
        return found == merge_pieces_.end() ? nullptr : &found->second;
    };

    MergeQueue candidates;
    ReverseUnusedMerges reverse_unused_merges;
    const auto add_candidate = [&](std::int32_t left, std::int32_t right) {
        if (left < 0 || right < 0) {
            return;
        }
        const auto& left_symbol = symbols[static_cast<std::size_t>(left)];
        const auto& right_symbol = symbols[static_cast<std::size_t>(right)];
        if (!left_symbol.alive || !right_symbol.alive || left_symbol.frozen ||
            right_symbol.frozen || left_symbol.next != right ||
            left_symbol.begin + left_symbol.size != right_symbol.begin) {
            return;
        }
        const std::string_view merged =
            std::string_view(normalized)
                .substr(left_symbol.begin, left_symbol.size + right_symbol.size);
        const PieceIndex* piece = lookup(merged);
        if (piece == nullptr) {
            return;
        }
        candidates.push({left, right, piece->score, merged.size()});
        if (piece->type == PieceType::kUnused) {
            // An unused piece is never emitted; remember the two halves so that the emitter can
            // put them back.
            reverse_unused_merges[std::string(merged)] = {
                std::string(symbol_text(normalized, symbols, left)),
                std::string(symbol_text(normalized, symbols, right))};
        }
    };

    for (std::size_t index = 1; index < symbols.size(); ++index) {
        add_candidate(static_cast<std::int32_t>(index - 1), static_cast<std::int32_t>(index));
    }
    while (!candidates.empty()) {
        const MergeCandidate candidate = candidates.top();
        candidates.pop();
        {
            // Lazy deletion: the stored size is what makes a candidate whose symbols have since
            // merged detectable.
            const auto& left = symbols[static_cast<std::size_t>(candidate.left)];
            const auto& right = symbols[static_cast<std::size_t>(candidate.right)];
            if (!left.alive || !right.alive || left.next != candidate.right ||
                left.size + right.size != candidate.size) {
                continue;
            }
        }
        auto& left = symbols[static_cast<std::size_t>(candidate.left)];
        auto& right = symbols[static_cast<std::size_t>(candidate.right)];
        left.size += right.size;
        left.next = right.next;
        if (right.next >= 0) {
            symbols[static_cast<std::size_t>(right.next)].previous = candidate.left;
        }
        right.alive = false;
        right.size = 0;
        add_candidate(left.previous, candidate.left);
        add_candidate(candidate.left, left.next);
    }

    std::vector<std::int32_t> ids;
    const auto emit = [&](std::string_view piece, auto&& self) -> void {
        const PieceIndex* found = lookup(piece);
        const std::int32_t id = found != nullptr ? found->id : asset_.unknown_id;
        const PieceType type = asset_.pieces[static_cast<std::size_t>(id)].type;
        // An unused piece is not a token the model has. If a merge recorded how it was built,
        // the two halves are emitted instead; with no such record SentencePiece treats the
        // symbol as unknown, so it goes down the same path here rather than emitting an id the
        // model never trained.
        bool unknown = type == PieceType::kUnknown;
        if (type == PieceType::kUnused) {
            const auto reverse = reverse_unused_merges.find(std::string(piece));
            if (reverse != reverse_unused_merges.end()) {
                self(reverse->second.first, self);
                self(reverse->second.second, self);
                return;
            }
            unknown = true;
        }
        if (unknown && asset_.byte_fallback) {
            for (const unsigned char byte : piece) {
                const std::int32_t byte_id = byte_ids_[byte];
                if (byte_id < 0) {
                    throw std::runtime_error("tipsv2 byte fallback table is incomplete");
                }
                ids.push_back(byte_id);
            }
            return;
        }
        ids.push_back(unknown ? asset_.unknown_id : id);
    };

    for (std::int32_t index = symbols.empty() ? -1 : 0; index >= 0;
         index = symbols[static_cast<std::size_t>(index)].next) {
        emit(symbol_text(normalized, symbols, index), emit);
    }
    // The checkpoint's tokenizer sets add_bos and add_eos to false, so nothing is appended here.
    return ids;
}

} // namespace trtmc::tipsv2
