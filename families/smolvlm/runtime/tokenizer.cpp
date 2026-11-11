/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tokenizer.h"

#include <array>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unordered_map>
#include <unordered_set>

namespace trtmc::smolvlm {
namespace {

// The GPT-2 ByteLevel split. "\s" is spelled out as the Unicode whitespace set
// of the reference regex engine (TAB..CR, NEL and every Z category), because
// ICU's own "\s" omits VT and NEL.
#define SMOLVLM_WS "\\t\\n\\x{0B}\\f\\r\\x{85}\\p{Z}"
constexpr const char* kByteLevelPattern =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^" SMOLVLM_WS "\\p{L}\\p{N}]+"
    "|[" SMOLVLM_WS "]+(?![^" SMOLVLM_WS "])|[" SMOLVLM_WS "]+";
#undef SMOLVLM_WS

std::string utf8(const icu::UnicodeString& text) {
    std::string result;
    text.toUTF8String(result);
    return result;
}

std::string pair_key(const std::string& left, const std::string& right) {
    return left + '\0' + right;
}

// Rust char::is_numeric: general categories Nd, Nl and No.
bool is_numeric(UChar32 cp) {
    const auto type = u_charType(cp);
    return type == U_DECIMAL_DIGIT_NUMBER || type == U_LETTER_NUMBER || type == U_OTHER_NUMBER;
}

} // namespace

struct Tokenizer::Impl {
    std::unique_ptr<icu::RegexPattern> pattern;
    std::unordered_map<std::string, int32_t> vocab, ranks;
    std::unordered_map<int32_t, std::string> inverse;
    std::unordered_set<int32_t> special_ids;
    std::vector<std::pair<std::string, int32_t>> added;
    std::array<std::string, 256> byte_encoder;
    std::array<bool, 256> byte_known{};
    std::unordered_map<UChar32, unsigned char> byte_decoder;

    explicit Impl(std::string_view serialized) {
        const auto document = nlohmann::json::parse(serialized);
        const auto& model = document.at("model");
        const auto& pre = document.at("pre_tokenizer");
        if (model.at("type") != "BPE" || !document.at("normalizer").is_null() ||
            !document.value("post_processor", nlohmann::json()).is_null() ||
            !model.value("dropout", nlohmann::json()).is_null() ||
            model.value("byte_fallback", false) || model.value("ignore_merges", false) ||
            !model.value("continuing_subword_prefix", nlohmann::json()).is_null() ||
            !model.value("end_of_word_suffix", nlohmann::json()).is_null())
            throw std::invalid_argument("Unsupported SmolVLM tokenizer model");
        const auto& parts = pre.at("pretokenizers");
        if (pre.at("type") != "Sequence" || parts.size() != 2 || parts[0].at("type") != "Digits" ||
            parts[0].at("individual_digits") != true || parts[1].at("type") != "ByteLevel" ||
            parts[1].at("add_prefix_space") != false || parts[1].at("use_regex") != true ||
            document.at("decoder").at("type") != "ByteLevel")
            throw std::invalid_argument("Unsupported SmolVLM pre-tokenizer");
        UErrorCode status = U_ZERO_ERROR;
        pattern.reset(
            icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(kByteLevelPattern), 0, status));
        if (U_FAILURE(status) || !pattern)
            throw std::runtime_error("SmolVLM ByteLevel split expression failed to compile");
        vocab = model.at("vocab").get<decltype(vocab)>();
        for (const auto& [token, id] : vocab)
            inverse.emplace(id, token);
        int32_t rank = 0;
        for (const auto& merge : model.at("merges")) {
            std::string left, right;
            if (merge.is_array() && merge.size() == 2) {
                left = merge.at(0);
                right = merge.at(1);
            } else {
                const auto text = merge.get<std::string>();
                const auto space = text.find(' ');
                if (space == std::string::npos || text.find(' ', space + 1) != std::string::npos)
                    throw std::invalid_argument("Malformed SmolVLM BPE merge");
                left = text.substr(0, space);
                right = text.substr(space + 1);
            }
            ranks.emplace(pair_key(left, right), rank++);
        }
        for (const auto& item : document.at("added_tokens")) {
            if (item.value("single_word", false) || item.value("lstrip", false) ||
                item.value("rstrip", false) || item.value("normalized", false))
                throw std::invalid_argument("SmolVLM added tokens must match literally");
            const std::string content = item.at("content");
            const int32_t id = item.at("id");
            if (content.empty())
                throw std::invalid_argument("Empty SmolVLM added token");
            added.emplace_back(content, id);
            inverse[id] = content;
            if (item.value("special", false))
                special_ids.insert(id);
        }
        int extra = 256;
        for (int value = 0; value < 256; ++value) {
            const bool literal =
                (value >= 33 && value <= 126) || (value >= 161 && value <= 172) || value >= 174;
            const UChar32 cp = literal ? value : extra++;
            byte_encoder[value] = utf8(icu::UnicodeString(cp));
            byte_decoder.emplace(cp, static_cast<unsigned char>(value));
        }
        for (int value = 0; value < 256; ++value)
            byte_known[value] = vocab.count(byte_encoder[value]) != 0;
    }

    void merge_piece(const std::string& bytes, std::vector<int32_t>& result) const {
        // Without an unknown token or byte fallback, a byte whose unit is not in
        // the vocabulary is dropped; its neighbours become adjacent.
        std::vector<std::string> units;
        units.reserve(bytes.size());
        for (unsigned char value : bytes)
            if (byte_known[value])
                units.push_back(byte_encoder[value]);
        // Merge the lowest-ranked adjacent pair first; the leftmost one on ties.
        while (units.size() > 1) {
            size_t best = units.size();
            int32_t best_rank = std::numeric_limits<int32_t>::max();
            for (size_t i = 0; i + 1 < units.size(); ++i) {
                const auto found = ranks.find(pair_key(units[i], units[i + 1]));
                if (found != ranks.end() && found->second < best_rank) {
                    best_rank = found->second;
                    best = i;
                }
            }
            if (best == units.size())
                break;
            units[best] += units[best + 1];
            units.erase(units.begin() + static_cast<std::ptrdiff_t>(best) + 1);
        }
        for (const auto& unit : units) {
            const auto found = vocab.find(unit);
            if (found == vocab.end())
                throw std::runtime_error("SmolVLM BPE produced a unit outside the vocabulary");
            result.push_back(found->second);
        }
    }

    void split_regex(const icu::UnicodeString& piece, icu::RegexMatcher& matcher,
                     std::vector<int32_t>& result) const {
        UErrorCode status = U_ZERO_ERROR;
        matcher.reset(piece);
        int32_t end = 0;
        while (matcher.find(status)) {
            const int32_t start = matcher.start(status);
            if (start > end)
                merge_piece(utf8(piece.tempSubStringBetween(end, start)), result);
            end = matcher.end(status);
            merge_piece(utf8(piece.tempSubStringBetween(start, end)), result);
        }
        if (U_FAILURE(status))
            throw std::runtime_error("SmolVLM ByteLevel split failed");
        if (end < piece.length())
            merge_piece(utf8(piece.tempSubString(end)), result);
    }

    // Digits(individual_digits) isolates every numeric code point, then the
    // ByteLevel expression splits each remaining run independently.
    void ordinary(std::string_view text, std::vector<int32_t>& result) const {
        if (text.empty())
            return;
        const auto input = icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), text.size()));
        UErrorCode status = U_ZERO_ERROR;
        std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(status));
        if (U_FAILURE(status) || !matcher)
            throw std::runtime_error("SmolVLM regex allocation failed");
        int32_t run = 0;
        for (int32_t i = 0; i < input.length();) {
            const UChar32 cp = input.char32At(i);
            const int32_t next = i + U16_LENGTH(cp);
            if (is_numeric(cp)) {
                if (i > run)
                    split_regex(input.tempSubStringBetween(run, i), *matcher, result);
                split_regex(input.tempSubStringBetween(i, next), *matcher, result);
                run = next;
            }
            i = next;
        }
        if (run < input.length())
            split_regex(input.tempSubString(run), *matcher, result);
    }
};

Tokenizer::Tokenizer(std::string_view serialized) : impl_(std::make_unique<Impl>(serialized)) {}
Tokenizer::~Tokenizer() = default;

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<int32_t> result;
    const auto& added = impl_->added;
    // Next occurrence of each added token; refreshed only once passed.
    std::vector<size_t> next(added.size());
    for (size_t i = 0; i < added.size(); ++i)
        next[i] = text.find(added[i].first);
    size_t offset = 0;
    while (offset < text.size()) {
        size_t first = std::string_view::npos, length = 0;
        int32_t id = -1;
        for (size_t i = 0; i < added.size(); ++i) {
            if (next[i] != std::string_view::npos && next[i] < offset)
                next[i] = text.find(added[i].first, offset);
            const size_t at = next[i];
            if (at != std::string_view::npos &&
                (at < first || (at == first && added[i].first.size() > length))) {
                first = at;
                length = added[i].first.size();
                id = added[i].second;
            }
        }
        if (first == std::string_view::npos) {
            impl_->ordinary(text.substr(offset), result);
            break;
        }
        impl_->ordinary(text.substr(offset, first - offset), result);
        result.push_back(id);
        offset = first + length;
    }
    return result;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string bytes;
    for (const int32_t id : ids) {
        if (impl_->special_ids.count(id))
            continue;
        const auto found = impl_->inverse.find(id);
        if (found == impl_->inverse.end())
            throw std::invalid_argument("SmolVLM token id is outside the vocabulary");
        const auto token = icu::UnicodeString::fromUTF8(found->second);
        for (int32_t i = 0; i < token.length();) {
            const UChar32 cp = token.char32At(i);
            i += U16_LENGTH(cp);
            const auto byte = impl_->byte_decoder.find(cp);
            if (byte == impl_->byte_decoder.end())
                bytes += utf8(icu::UnicodeString(cp));
            else
                bytes.push_back(static_cast<char>(byte->second));
        }
    }
    return utf8(icu::UnicodeString::fromUTF8(bytes));
}

int32_t Tokenizer::added_token_id(std::string_view content) const {
    for (const auto& [text, id] : impl_->added)
        if (text == content)
            return id;
    return -1;
}

} // namespace trtmc::smolvlm
