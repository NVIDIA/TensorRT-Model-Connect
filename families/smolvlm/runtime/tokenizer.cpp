/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tokenizer.h"

#include <array>
#include <functional>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <tuple>
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

uint64_t id_pair(int32_t left, int32_t right) {
    return (uint64_t(uint32_t(left)) << 32) | uint32_t(right);
}

// Rust char::is_numeric: general categories Nd, Nl and No.
bool is_numeric(UChar32 cp) {
    const auto type = u_charType(cp);
    return type == U_DECIMAL_DIGIT_NUMBER || type == U_LETTER_NUMBER || type == U_OTHER_NUMBER;
}

} // namespace

struct Tokenizer::Impl {
    struct Merge {
        int32_t rank, id;
    };

    std::unique_ptr<icu::RegexPattern> pattern;
    std::unordered_map<std::string, int32_t> vocab;
    // (left id, right id) -> rank in the merges list and the id of the merged unit.
    std::unordered_map<uint64_t, Merge> merges;
    std::unordered_map<int32_t, std::string> inverse;
    std::unordered_set<int32_t> special_ids;
    std::vector<std::pair<std::string, int32_t>> added;
    std::array<std::string, 256> byte_encoder;
    // Vocabulary id of each byte's unit, -1 when the vocabulary lacks it.
    std::array<int32_t, 256> byte_id{};
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
            const auto left_id = vocab.find(left), right_id = vocab.find(right),
                       merged_id = vocab.find(left + right);
            if (left_id == vocab.end() || right_id == vocab.end() || merged_id == vocab.end())
                throw std::invalid_argument("SmolVLM BPE merge outside the vocabulary");
            merges.emplace(id_pair(left_id->second, right_id->second),
                           Merge{rank++, merged_id->second});
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
        for (int value = 0; value < 256; ++value) {
            const auto found = vocab.find(byte_encoder[value]);
            byte_id[value] = found == vocab.end() ? -1 : found->second;
        }
    }

    // Merges the lowest-ranked adjacent pair first, the leftmost one on ties, until no
    // pair has a merge. As in the reference (tokenizers' Word::merge_all), candidate
    // merges wait in a min-heap keyed by (rank, position) over a linked list of units,
    // so a piece of n bytes takes O(n log n) instead of a full rescan per merge.
    void merge_piece(const std::string& bytes, std::vector<int32_t>& result) const {
        struct Unit {
            int32_t id, prev, next; // prev / next: neighbouring unit, -1 at the ends
            bool live;
        };
        // Without an unknown token or byte fallback, a byte whose unit is not in
        // the vocabulary is dropped; its neighbours become adjacent.
        std::vector<Unit> units;
        units.reserve(bytes.size());
        for (unsigned char value : bytes) {
            if (byte_id[value] < 0)
                continue;
            const auto index = static_cast<int32_t>(units.size());
            units.push_back({byte_id[value], index - 1, index + 1, true});
        }
        if (units.empty())
            return;
        units.back().next = -1;
        using Candidate = std::tuple<int32_t, int32_t, int32_t>; // rank, left unit, merged id
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue;
        const auto offer = [&](int32_t left) {
            const int32_t right = units[size_t(left)].next;
            if (right < 0)
                return;
            const auto found =
                merges.find(id_pair(units[size_t(left)].id, units[size_t(right)].id));
            if (found != merges.end())
                queue.emplace(found->second.rank, left, found->second.id);
        };
        for (int32_t index = 0; index + 1 < static_cast<int32_t>(units.size()); ++index)
            offer(index);
        while (!queue.empty()) {
            const int32_t left = std::get<1>(queue.top()), merged = std::get<2>(queue.top());
            queue.pop();
            Unit& unit = units[size_t(left)];
            // An entry is stale when an earlier merge removed its unit or changed its pair.
            if (!unit.live || unit.next < 0)
                continue;
            Unit& right = units[size_t(unit.next)];
            const auto found = merges.find(id_pair(unit.id, right.id));
            if (found == merges.end() || found->second.id != merged)
                continue;
            unit.id = merged;
            right.live = false;
            unit.next = right.next;
            if (right.next >= 0)
                units[size_t(right.next)].prev = left;
            if (unit.prev >= 0)
                offer(unit.prev);
            offer(left);
        }
        // The first unit only ever absorbs its right neighbours, so it heads the list.
        for (int32_t index = 0; index >= 0; index = units[size_t(index)].next)
            result.push_back(units[size_t(index)].id);
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
