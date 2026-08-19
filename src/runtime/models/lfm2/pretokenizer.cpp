/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lfm2/pretokenizer.h"

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

Utf8Unit decode_utf8_unit(std::string_view text, std::size_t offset) {
    const auto first = static_cast<unsigned char>(text[offset]);
    if (first < 0x80U)
        return {first, 1};
    if (first >= 0xC2U && first <= 0xDFU && offset + 1 < text.size()) {
        const auto second = static_cast<unsigned char>(text[offset + 1]);
        if (is_continuation(second)) {
            return {static_cast<char32_t>((first & 0x1FU) << 6U) |
                        static_cast<char32_t>(second & 0x3FU),
                    2};
        }
    }
    if (first >= 0xE0U && first <= 0xEFU && offset + 2 < text.size()) {
        const auto second = static_cast<unsigned char>(text[offset + 1]);
        const auto third = static_cast<unsigned char>(text[offset + 2]);
        const bool scalar_prefix =
            (first != 0xE0U || second >= 0xA0U) && (first != 0xEDU || second <= 0x9FU);
        if (scalar_prefix && is_continuation(second) && is_continuation(third)) {
            return {static_cast<char32_t>((first & 0x0FU) << 12U) |
                        static_cast<char32_t>((second & 0x3FU) << 6U) |
                        static_cast<char32_t>(third & 0x3FU),
                    3};
        }
    }
    if (first >= 0xF0U && first <= 0xF4U && offset + 3 < text.size()) {
        const auto second = static_cast<unsigned char>(text[offset + 1]);
        const auto third = static_cast<unsigned char>(text[offset + 2]);
        const auto fourth = static_cast<unsigned char>(text[offset + 3]);
        const bool scalar_prefix =
            (first != 0xF0U || second >= 0x90U) && (first != 0xF4U || second <= 0x8FU);
        if (scalar_prefix && is_continuation(second) && is_continuation(third) &&
            is_continuation(fourth)) {
            return {static_cast<char32_t>((first & 0x07U) << 18U) |
                        static_cast<char32_t>((second & 0x3FU) << 12U) |
                        static_cast<char32_t>((third & 0x3FU) << 6U) |
                        static_cast<char32_t>(fourth & 0x3FU),
                    4};
        }
    }
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

class Lfm2PinnedPretokenizer final : public ITokenizer {
  public:
    Lfm2PinnedPretokenizer(std::unique_ptr<ITokenizer> inner, std::vector<std::string> added_tokens)
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
        for (const auto& piece : lfm2_split_pretokens(text))
            append_inner(piece, ids);
    }

    std::unique_ptr<ITokenizer> inner_;
    std::vector<std::string> added_tokens_;
};

} // namespace

bool lfm2_uses_pinned_split_byte_level_pretokenizer(const char* tokenizer_json, std::size_t size) {
    if (tokenizer_json == nullptr || size == 0)
        return false;
    try {
        const auto root = nlohmann::json::parse(tokenizer_json, tokenizer_json + size);
        const auto pre = root.find("pre_tokenizer");
        if (pre == root.end() || !pre->is_object() || pre->value("type", "") != "Sequence")
            return false;
        const auto children = pre->find("pretokenizers");
        if (children == pre->end() || !children->is_array() || children->size() != 2)
            return false;
        const auto& split = (*children)[0];
        const auto& byte_level = (*children)[1];
        return split.is_object() && split.value("type", "") == "Split" &&
               split.value("behavior", "") == "Isolated" && !split.value("invert", true) &&
               split.contains("pattern") && split["pattern"].is_object() &&
               split["pattern"].value("Regex", "") == kPinnedSplitRegex && byte_level.is_object() &&
               byte_level.value("type", "") == "ByteLevel" &&
               !byte_level.value("add_prefix_space", true) && !byte_level.value("use_regex", true);
    } catch (const nlohmann::json::exception&) {
        return false;
    }
}

std::vector<std::string> lfm2_tokenizer_added_tokens(const char* tokenizer_json, std::size_t size) {
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

std::vector<std::string> lfm2_split_pretokens(std::string_view text) {
    std::vector<std::string> pieces;
    for (std::size_t offset = 0; offset < text.size();) {
        const std::size_t begin = offset;

        if (const std::size_t size = contraction_size(text, offset); size > 0) {
            offset += size;
            append_piece(text, begin, offset, pieces);
            continue;
        }

        const auto first = decode_utf8_unit(text, offset);
        if (is_letter(first.codepoint)) {
            offset = consume_while(text, offset, is_letter);
            append_piece(text, begin, offset, pieces);
            continue;
        }
        if (!is_newline(first.codepoint) && !is_letter(first.codepoint) &&
            !is_number(first.codepoint)) {
            const std::size_t after_prefix = offset + first.size;
            if (after_prefix < text.size() &&
                is_letter(decode_utf8_unit(text, after_prefix).codepoint)) {
                offset = consume_while(text, after_prefix, is_letter);
                append_piece(text, begin, offset, pieces);
                continue;
            }
        }

        if (is_number(first.codepoint)) {
            offset = consume_while(text, offset, is_number, 3);
            append_piece(text, begin, offset, pieces);
            continue;
        }

        std::size_t other_begin = offset;
        if (first.codepoint == ' ')
            other_begin += first.size;
        if (other_begin < text.size() && is_other(decode_utf8_unit(text, other_begin).codepoint)) {
            offset = consume_while(text, other_begin, is_other);
            offset = consume_while(text, offset, is_newline);
            append_piece(text, begin, offset, pieces);
            continue;
        }

        if (is_whitespace(first.codepoint)) {
            std::size_t scan = offset;
            std::size_t last_newline_end = offset;
            while (scan < text.size()) {
                const auto unit = decode_utf8_unit(text, scan);
                if (!is_whitespace(unit.codepoint))
                    break;
                scan += unit.size;
                if (is_newline(unit.codepoint))
                    last_newline_end = scan;
            }
            offset = last_newline_end > begin ? last_newline_end : scan;
            append_piece(text, begin, offset, pieces);
            continue;
        }

        offset += first.size;
        append_piece(text, begin, offset, pieces);
    }
    return pieces;
}

std::unique_ptr<ITokenizer> lfm2_wrap_pinned_pretokenizer(std::unique_ptr<ITokenizer> tokenizer,
                                                          std::vector<std::string> added_tokens) {
    return std::make_unique<Lfm2PinnedPretokenizer>(std::move(tokenizer), std::move(added_tokens));
}

} // namespace trtmc
