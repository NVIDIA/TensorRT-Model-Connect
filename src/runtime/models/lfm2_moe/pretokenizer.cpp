/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lfm2_moe/pretokenizer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

constexpr std::string_view kPinnedSplitRegex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

struct Utf8Unit {
    char32_t codepoint{0};
    std::size_t size{1};
};

bool is_continuation(unsigned char byte) {
    return (byte & 0xC0U) == 0x80U;
}

std::size_t utf8_sequence_size(unsigned char first) {
    if (first < 0x80U)
        return 1;
    if (first >= 0xC2U && first <= 0xDFU)
        return 2;
    if (first >= 0xE0U && first <= 0xEFU)
        return 3;
    if (first >= 0xF0U && first <= 0xF4U)
        return 4;
    return 0;
}

bool is_valid_three_byte_prefix(unsigned char first, unsigned char second) {
    return (first != 0xE0U || second >= 0xA0U) && (first != 0xEDU || second <= 0x9FU);
}

bool is_valid_four_byte_prefix(unsigned char first, unsigned char second) {
    return (first != 0xF0U || second >= 0x90U) && (first != 0xF4U || second <= 0x8FU);
}

bool is_valid_scalar_prefix(unsigned char first, unsigned char second, std::size_t size) {
    if (size == 3)
        return is_valid_three_byte_prefix(first, second);
    if (size == 4)
        return is_valid_four_byte_prefix(first, second);
    return true;
}

bool is_valid_utf8_sequence(std::string_view text, std::size_t offset, std::size_t size,
                            unsigned char first) {
    if (size == 0 || size > text.size() - offset)
        return false;
    if (size > 1 &&
        !is_valid_scalar_prefix(first, static_cast<unsigned char>(text[offset + 1]), size))
        return false;
    for (std::size_t index = 1; index < size; ++index) {
        if (!is_continuation(static_cast<unsigned char>(text[offset + index])))
            return false;
    }
    return true;
}

char32_t decode_utf8_codepoint(std::string_view text, std::size_t offset, std::size_t size) {
    static constexpr std::array<unsigned char, 5> masks{0, 0x7FU, 0x1FU, 0x0FU, 0x07U};
    char32_t codepoint = static_cast<unsigned char>(text[offset]) & masks[size];
    for (std::size_t index = 1; index < size; ++index) {
        codepoint = (codepoint << 6U) | (static_cast<unsigned char>(text[offset + index]) & 0x3FU);
    }
    return codepoint;
}

Utf8Unit decode_utf8_unit(std::string_view text, std::size_t offset) {
    const auto first = static_cast<unsigned char>(text[offset]);
    const std::size_t size = utf8_sequence_size(first);
    if (is_valid_utf8_sequence(text, offset, size, first))
        return {decode_utf8_codepoint(text, offset, size), size};
    return {first, 1};
}

struct Range {
    char32_t first;
    char32_t last;
};

template <std::size_t N>
bool in_ranges(char32_t codepoint, const std::array<Range, N>& ranges) {
    return std::any_of(ranges.begin(), ranges.end(), [codepoint](const Range& range) {
        return codepoint >= range.first && codepoint <= range.last;
    });
}

bool is_letter(char32_t codepoint) {
    // Unicode Letter categories needed by every language advertised for the
    // pinned LFM2 checkpoint. Deliberate holes exclude punctuation and marks:
    // notably Arabic comma U+060C and Katakana middle dot U+30FB.
    static constexpr std::array<Range, 28> ranges{{
        {'A', 'Z'},       {'a', 'z'},       {0x00AA, 0x00AA}, {0x00B5, 0x00B5}, {0x00BA, 0x00BA},
        {0x00C0, 0x00D6}, {0x00D8, 0x00F6}, {0x00F8, 0x02C1}, {0x0400, 0x0481}, {0x048A, 0x052F},
        {0x0620, 0x064A}, {0x066E, 0x066F}, {0x0671, 0x06D3}, {0x06E5, 0x06E6}, {0x06EE, 0x06EF},
        {0x06FA, 0x06FC}, {0x06FF, 0x06FF}, {0x0904, 0x0939}, {0x093D, 0x093D}, {0x0950, 0x0950},
        {0x0958, 0x0961}, {0x3041, 0x3096}, {0x309D, 0x309F}, {0x30A1, 0x30FA}, {0x30FC, 0x30FF},
        {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xAC00, 0xD7A3},
    }};
    return in_ranges(codepoint, ranges);
}

bool is_number(char32_t codepoint) {
    static constexpr std::array<Range, 6> ranges{{
        {'0', '9'},
        {0x0660, 0x0669},
        {0x06F0, 0x06F9},
        {0x0966, 0x096F},
        {0xFF10, 0xFF19},
        {0x1D7CE, 0x1D7FF},
    }};
    return in_ranges(codepoint, ranges);
}

bool is_whitespace(char32_t codepoint) {
    static constexpr std::array<char32_t, 14> singletons{
        ' ',  '\t',   '\n',   '\r',   0x0B,   0x0C,   0x85,
        0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
    };
    return (codepoint >= 0x2000 && codepoint <= 0x200A) ||
           std::find(singletons.begin(), singletons.end(), codepoint) != singletons.end();
}

bool is_newline(char32_t codepoint) {
    return codepoint == '\r' || codepoint == '\n';
}

bool is_other(char32_t codepoint) {
    return !is_letter(codepoint) && !is_number(codepoint) && !is_whitespace(codepoint);
}

char ascii_lower(char value) {
    return value >= 'A' && value <= 'Z' ? static_cast<char>(value - 'A' + 'a') : value;
}

std::size_t contraction_size(std::string_view text, std::size_t offset) {
    if (text[offset] != '\'')
        return 0;
    static constexpr std::array<std::string_view, 7> suffixes{"re", "ve", "ll", "s", "t", "m", "d"};
    for (const auto suffix : suffixes) {
        if (offset + 1 + suffix.size() > text.size())
            continue;
        bool matches = true;
        for (std::size_t index = 0; index < suffix.size(); ++index) {
            if (ascii_lower(text[offset + 1 + index]) != suffix[index]) {
                matches = false;
                break;
            }
        }
        if (matches)
            return 1 + suffix.size();
    }
    return 0;
}

std::size_t consume_while(std::string_view text, std::size_t offset, bool (*predicate)(char32_t),
                          std::size_t maximum = SIZE_MAX) {
    std::size_t count = 0;
    while (offset < text.size() && count < maximum) {
        const auto unit = decode_utf8_unit(text, offset);
        if (!predicate(unit.codepoint))
            break;
        offset += unit.size;
        ++count;
    }
    return offset;
}

void append_piece(std::string_view text, std::size_t begin, std::size_t end,
                  std::vector<std::string>& pieces) {
    if (end > begin)
        pieces.emplace_back(text.substr(begin, end - begin));
}

std::size_t contraction_piece_end(std::string_view text, std::size_t offset) {
    const std::size_t size = contraction_size(text, offset);
    return size == 0 ? offset : offset + size;
}

std::size_t letter_piece_end(std::string_view text, std::size_t offset, const Utf8Unit& first) {
    if (is_letter(first.codepoint))
        return consume_while(text, offset, is_letter);
    if (is_newline(first.codepoint) || is_number(first.codepoint))
        return offset;

    const std::size_t after_prefix = offset + first.size;
    if (after_prefix >= text.size())
        return offset;
    if (!is_letter(decode_utf8_unit(text, after_prefix).codepoint))
        return offset;
    return consume_while(text, after_prefix, is_letter);
}

std::size_t number_piece_end(std::string_view text, std::size_t offset, const Utf8Unit& first) {
    if (!is_number(first.codepoint))
        return offset;
    return consume_while(text, offset, is_number, 3);
}

std::size_t other_piece_end(std::string_view text, std::size_t offset, const Utf8Unit& first) {
    std::size_t other_begin = offset;
    if (first.codepoint == ' ')
        other_begin += first.size;
    if (other_begin >= text.size())
        return offset;
    if (!is_other(decode_utf8_unit(text, other_begin).codepoint))
        return offset;

    const std::size_t other_end = consume_while(text, other_begin, is_other);
    return consume_while(text, other_end, is_newline);
}

std::size_t whitespace_piece_end(std::string_view text, std::size_t offset, const Utf8Unit& first) {
    if (!is_whitespace(first.codepoint))
        return offset;

    std::size_t scan = offset;
    std::size_t last_newline_end = offset;
    std::size_t last_unit_begin = offset;
    while (scan < text.size()) {
        const auto unit = decode_utf8_unit(text, scan);
        if (!is_whitespace(unit.codepoint))
            break;
        last_unit_begin = scan;
        scan += unit.size;
        if (is_newline(unit.codepoint))
            last_newline_end = scan;
    }
    if (last_newline_end > offset)
        return last_newline_end;
    // \s+(?!\S): a newline-free run followed by non-whitespace yields its
    // final whitespace unit to the next piece (it becomes that piece's
    // prefix). A single-unit run has nothing left to yield and falls through
    // to the plain \s+ alternative, consuming the unit whole.
    if (scan < text.size() && last_unit_begin > offset)
        return last_unit_begin;
    return scan;
}

std::size_t next_pretoken_end(std::string_view text, std::size_t offset) {
    std::size_t end = contraction_piece_end(text, offset);
    if (end != offset)
        return end;

    const auto first = decode_utf8_unit(text, offset);
    end = letter_piece_end(text, offset, first);
    if (end != offset)
        return end;
    end = number_piece_end(text, offset, first);
    if (end != offset)
        return end;
    end = other_piece_end(text, offset, first);
    if (end != offset)
        return end;
    end = whitespace_piece_end(text, offset, first);
    if (end != offset)
        return end;
    return offset + first.size;
}

class Lfm2MoePinnedPretokenizer final : public ITokenizer {
  public:
    Lfm2MoePinnedPretokenizer(std::unique_ptr<ITokenizer> inner,
                              std::vector<std::string> added_tokens)
        : inner_(std::move(inner)), added_tokens_(std::move(added_tokens)) {
        if (!inner_)
            throw std::invalid_argument("LFM2 pretokenizer requires an inner tokenizer");
        added_tokens_.erase(std::remove_if(added_tokens_.begin(), added_tokens_.end(),
                                           [](const std::string& token) { return token.empty(); }),
                            added_tokens_.end());
        std::sort(
            added_tokens_.begin(), added_tokens_.end(),
            [](const std::string& lhs, const std::string& rhs) { return lhs.size() > rhs.size(); });
        added_tokens_.erase(std::unique(added_tokens_.begin(), added_tokens_.end()),
                            added_tokens_.end());
    }

    std::vector<int32_t> encode(const std::string& text) const override {
        std::vector<int32_t> ids;
        std::size_t normal_begin = 0;
        for (std::size_t offset = 0; offset < text.size();) {
            const std::string* matched = nullptr;
            for (const auto& token : added_tokens_) {
                if (token.front() != text[offset])
                    continue;
                if (text.compare(offset, token.size(), token) == 0) {
                    matched = &token;
                    break;
                }
            }
            if (matched == nullptr) {
                ++offset;
                continue;
            }
            append_normal(std::string_view(text).substr(normal_begin, offset - normal_begin), ids);
            append_inner(*matched, ids);
            offset += matched->size();
            normal_begin = offset;
        }
        append_normal(std::string_view(text).substr(normal_begin), ids);
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        return inner_->decode(ids);
    }
    int32_t id_for_token(std::string_view token) const override {
        return inner_->id_for_token(token);
    }
    std::string token_for_id(int32_t id) const override { return inner_->token_for_id(id); }

  private:
    void append_inner(const std::string& piece, std::vector<int32_t>& ids) const {
        // The shared BPE's internal contraction scanner is lowercase-only. The
        // pinned regex is case-insensitive, and the pinned vocabulary contains
        // complete uppercase one-letter contractions such as 'M, 'T, 'S, and
        // 'D. Select those exact byte-level tokens before delegating; longer
        // contractions ('RE/'LL/'VE) intentionally fall through to BPE.
        if (piece.size() == 2 && piece[0] == '\'' && piece[1] >= 'A' && piece[1] <= 'Z' &&
            contraction_size(piece, 0) == piece.size()) {
            const int32_t token_id = inner_->id_for_token(piece);
            if (token_id >= 0) {
                ids.push_back(token_id);
                return;
            }
        }
        const auto encoded = inner_->encode(piece);
        ids.insert(ids.end(), encoded.begin(), encoded.end());
    }

    void append_normal(std::string_view text, std::vector<int32_t>& ids) const {
        for (const auto& piece : lfm2_moe_split_pretokens(text))
            append_inner(piece, ids);
    }

    std::unique_ptr<ITokenizer> inner_;
    std::vector<std::string> added_tokens_;
};

bool json_has_type(const nlohmann::json& value, std::string_view type) {
    return value.is_object() && value.value("type", std::string{}) == type;
}

bool is_pinned_split_pattern(const nlohmann::json& split) {
    const auto pattern = split.find("pattern");
    if (pattern == split.end() || !pattern->is_object())
        return false;
    return pattern->value("Regex", std::string{}) == kPinnedSplitRegex;
}

bool is_pinned_split(const nlohmann::json& split) {
    if (!json_has_type(split, "Split"))
        return false;
    if (split.value("behavior", std::string{}) != "Isolated")
        return false;
    if (split.value("invert", true))
        return false;
    return is_pinned_split_pattern(split);
}

bool is_pinned_byte_level(const nlohmann::json& byte_level) {
    if (!json_has_type(byte_level, "ByteLevel"))
        return false;
    if (byte_level.value("add_prefix_space", true))
        return false;
    return !byte_level.value("use_regex", true);
}

bool is_pinned_pretokenizer(const nlohmann::json& pretokenizer) {
    if (!json_has_type(pretokenizer, "Sequence"))
        return false;
    const auto children = pretokenizer.find("pretokenizers");
    if (children == pretokenizer.end() || !children->is_array() || children->size() != 2)
        return false;
    return is_pinned_split((*children)[0]) && is_pinned_byte_level((*children)[1]);
}

} // namespace

bool lfm2_moe_uses_pinned_split_byte_level_pretokenizer(const char* tokenizer_json,
                                                        std::size_t size) {
    if (tokenizer_json == nullptr || size == 0)
        return false;
    try {
        const auto root = nlohmann::json::parse(tokenizer_json, tokenizer_json + size);
        const auto pre = root.find("pre_tokenizer");
        return pre != root.end() && is_pinned_pretokenizer(*pre);
    } catch (const nlohmann::json::exception&) {
        return false;
    }
}

std::vector<std::string> lfm2_moe_tokenizer_added_tokens(const char* tokenizer_json,
                                                         std::size_t size) {
    std::vector<std::string> tokens;
    if (tokenizer_json == nullptr || size == 0)
        return tokens;
    try {
        const auto root = nlohmann::json::parse(tokenizer_json, tokenizer_json + size);
        const auto added = root.find("added_tokens");
        if (added == root.end() || !added->is_array())
            return tokens;
        for (const auto& entry : *added) {
            if (entry.is_object() && entry.contains("content") && entry["content"].is_string())
                tokens.push_back(entry["content"].get<std::string>());
        }
    } catch (const nlohmann::json::exception&) {
        tokens.clear();
    }
    return tokens;
}

std::vector<std::string> lfm2_moe_split_pretokens(std::string_view text) {
    std::vector<std::string> pieces;
    for (std::size_t offset = 0; offset < text.size();) {
        const std::size_t begin = offset;
        offset = next_pretoken_end(text, offset);
        append_piece(text, begin, offset, pieces);
    }
    return pieces;
}

std::unique_ptr<ITokenizer>
lfm2_moe_wrap_pinned_pretokenizer(std::unique_ptr<ITokenizer> tokenizer,
                                  std::vector<std::string> added_tokens) {
    return std::make_unique<Lfm2MoePinnedPretokenizer>(std::move(tokenizer),
                                                       std::move(added_tokens));
}

} // namespace trtmc
