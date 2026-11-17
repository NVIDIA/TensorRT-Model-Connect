/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/layoutlm/runtime/tokenizer.h"

#include "families/layoutlm/runtime/utf8.h"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <unicode/uchar.h>
#include <utility>

namespace trtmc::layoutlm {
namespace {

// GPT-2 byte-to-Unicode map: printable bytes stand for themselves, the other
// bytes are shifted to U+0100 onwards in increasing byte order.
std::string byte_symbol(int byte) {
    const auto direct = [](int value) {
        return (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
               (value >= 174 && value <= 255);
    };
    char32_t value = static_cast<char32_t>(byte);
    if (!direct(byte)) {
        int shifted = 0;
        for (int other = 0; other < byte; ++other)
            shifted += direct(other) ? 0 : 1;
        value = static_cast<char32_t>(256 + shifted);
    }
    std::string utf8;
    if (value < 0x80) {
        utf8.push_back(static_cast<char>(value));
    } else {
        utf8.push_back(static_cast<char>(0xC0 | (value >> 6)));
        utf8.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    }
    return utf8;
}

// Character classes of the GPT-2 pre-tokenizer regex
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// with \s = Unicode White_Space (the Oniguruma class the reference uses).
enum class CharClass { kLetter, kNumber, kSpace, kOther };

CharClass classify(char32_t value) {
    const auto category = U_MASK(u_charType(static_cast<UChar32>(value)));
    if (category & U_GC_L_MASK)
        return CharClass::kLetter;
    if (category & U_GC_N_MASK)
        return CharClass::kNumber;
    if (u_hasBinaryProperty(static_cast<UChar32>(value), UCHAR_WHITE_SPACE))
        return CharClass::kSpace;
    return CharClass::kOther;
}

std::size_t contraction_length(const std::vector<CodePoint>& points, std::size_t index) {
    if (points[index].value != U'\'' || index + 1 >= points.size())
        return 0;
    const char32_t first = points[index + 1].value;
    if (first == U's' || first == U't' || first == U'm' || first == U'd')
        return 2;
    if (index + 2 >= points.size())
        return 0;
    const char32_t second = points[index + 2].value;
    if ((first == U'r' && second == U'e') || (first == U'v' && second == U'e') ||
        (first == U'l' && second == U'l'))
        return 3;
    return 0;
}

// Leftmost-first alternation of the regex above, one match at a time.
std::vector<std::string_view> pre_tokenize(std::string_view text) {
    const auto points = decode_utf8(text);
    std::vector<CharClass> classes;
    classes.reserve(points.size());
    for (const auto& point : points)
        classes.push_back(classify(point.value));

    std::vector<std::string_view> pieces;
    const std::size_t count = points.size();
    std::size_t index = 0;
    while (index < count) {
        std::size_t end = index + contraction_length(points, index);
        if (end == index) {
            const bool spaced_run = points[index].value == U' ' && index + 1 < count &&
                                    classes[index + 1] != CharClass::kSpace;
            if (spaced_run || classes[index] != CharClass::kSpace) {
                const std::size_t first = spaced_run ? index + 1 : index;
                end = first + 1;
                while (end < count && classes[end] == classes[first])
                    ++end;
            } else {
                end = index + 1;
                while (end < count && classes[end] == CharClass::kSpace)
                    ++end;
                // \s+(?!\S): a run followed by a non-space leaves its last
                // character to prefix the next match; a single one stays \s+.
                if (end < count && end - index >= 2)
                    --end;
            }
        }
        pieces.push_back(
            text.substr(points[index].begin, points[end - 1].end - points[index].begin));
        index = end;
    }
    return pieces;
}

bool is_white_space(char32_t value) {
    return u_hasBinaryProperty(static_cast<UChar32>(value), UCHAR_WHITE_SPACE);
}

std::uint64_t pair_key(std::int32_t left, std::int32_t right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
           static_cast<std::uint32_t>(right);
}

} // namespace

std::unique_ptr<WordTokenizer> WordTokenizer::create(std::string_view tokenizer_json) {
    const auto json = nlohmann::json::parse(tokenizer_json.begin(), tokenizer_json.end());
    const auto& model = json.at("model");
    if (model.at("type").get<std::string>() != "BPE")
        throw std::runtime_error("layoutlm tokenizer.json is not a BPE model");

    auto tokenizer = std::unique_ptr<WordTokenizer>(new WordTokenizer());
    std::int32_t max_id = -1;
    for (const auto& [token, id] : model.at("vocab").items()) {
        const auto value = id.get<std::int32_t>();
        if (value < 0)
            throw std::runtime_error("layoutlm tokenizer.json has a negative token id");
        tokenizer->vocab_.emplace(token, value);
        max_id = std::max(max_id, value);
    }
    for (const auto& added : json.at("added_tokens")) {
        AddedToken token;
        token.content = added.at("content").get<std::string>();
        token.id = added.at("id").get<std::int32_t>();
        token.lstrip = added.value("lstrip", false);
        token.rstrip = added.value("rstrip", false);
        if (token.content.empty() || token.id < 0 || added.value("single_word", false) ||
            added.value("normalized", false))
            throw std::runtime_error("layoutlm tokenizer.json has an unsupported added token");
        max_id = std::max(max_id, token.id);
        tokenizer->added_.push_back(std::move(token));
    }
    tokenizer->vocab_size_ = static_cast<std::size_t>(max_id) + 1;
    tokenizer->tokens_.resize(tokenizer->vocab_size_);
    for (const auto& [token, id] : tokenizer->vocab_)
        tokenizer->tokens_[static_cast<std::size_t>(id)] = token;

    for (int byte = 0; byte < 256; ++byte) {
        const auto found = tokenizer->vocab_.find(byte_symbol(byte));
        if (found == tokenizer->vocab_.end())
            throw std::runtime_error("layoutlm tokenizer.json lacks a byte-level symbol");
        tokenizer->byte_ids_[byte] = found->second;
    }

    const auto& merges = model.at("merges");
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        std::string left;
        std::string right;
        if (merges[rank].is_string()) {
            const auto line = merges[rank].get<std::string>();
            const auto space = line.find(' ');
            if (space == std::string::npos || line.find(' ', space + 1) != std::string::npos)
                throw std::runtime_error("layoutlm tokenizer.json has a malformed merge");
            left = line.substr(0, space);
            right = line.substr(space + 1);
        } else {
            left = merges[rank].at(0).get<std::string>();
            right = merges[rank].at(1).get<std::string>();
        }
        const auto left_id = tokenizer->vocab_.find(left);
        const auto right_id = tokenizer->vocab_.find(right);
        const auto merged = tokenizer->vocab_.find(left + right);
        if (left_id == tokenizer->vocab_.end() || right_id == tokenizer->vocab_.end() ||
            merged == tokenizer->vocab_.end())
            throw std::runtime_error(
                "layoutlm tokenizer.json merges a token outside the vocabulary");
        // A repeated pair keeps its last rank, as the reference's map does.
        tokenizer->merges_[pair_key(left_id->second, right_id->second)] = {
            static_cast<std::int32_t>(rank), merged->second};
    }

    const auto& processor = json.at("post_processor");
    if (processor.at("type").get<std::string>() != "RobertaProcessing")
        throw std::runtime_error(
            "layoutlm tokenizer.json does not frame pairs as <s> A </s></s> B </s>");
    tokenizer->cls_id_ = processor.at("cls").at(1).get<std::int32_t>();
    tokenizer->sep_id_ = processor.at("sep").at(1).get<std::int32_t>();
    if (tokenizer->cls_id_ < 0 || tokenizer->sep_id_ < 0 ||
        static_cast<std::size_t>(tokenizer->cls_id_) >= tokenizer->vocab_size_ ||
        static_cast<std::size_t>(tokenizer->sep_id_) >= tokenizer->vocab_size_)
        throw std::runtime_error("layoutlm tokenizer.json frames with ids outside the vocabulary");
    return tokenizer;
}

const WordTokenizer::Merge* WordTokenizer::find_merge(std::int32_t left, std::int32_t right) const {
    const auto found = merges_.find(pair_key(left, right));
    return found == merges_.end() ? nullptr : &found->second;
}

void WordTokenizer::encode_pre_token(std::string_view pre_token,
                                     std::vector<std::int32_t>& ids) const {
    struct Symbol {
        std::int32_t id;
        std::int64_t previous;
        std::int64_t next;
        bool removed;
    };
    std::vector<Symbol> symbols;
    symbols.reserve(pre_token.size());
    for (std::size_t index = 0; index < pre_token.size(); ++index) {
        symbols.push_back({byte_ids_[static_cast<unsigned char>(pre_token[index])],
                           static_cast<std::int64_t>(index) - 1,
                           index + 1 < pre_token.size() ? static_cast<std::int64_t>(index) + 1 : -1,
                           false});
    }

    // Lowest rank first, then leftmost: the reference's merge queue order.
    struct Candidate {
        std::int32_t rank;
        std::size_t position;
        std::int32_t new_id;
    };
    const auto later = [](const Candidate& left, const Candidate& right) {
        return left.rank != right.rank ? left.rank > right.rank : left.position > right.position;
    };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)> queue(later);
    for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
        if (const auto* merge = find_merge(symbols[index].id, symbols[index + 1].id))
            queue.push({merge->rank, index, merge->new_id});
    }
    while (!queue.empty()) {
        const auto top = queue.top();
        queue.pop();
        auto& current = symbols[top.position];
        if (current.removed || current.next < 0)
            continue;
        const auto next_position = static_cast<std::size_t>(current.next);
        const auto* merge = find_merge(current.id, symbols[next_position].id);
        // An entry made stale by an earlier merge no longer names this pair.
        if (merge == nullptr || merge->new_id != top.new_id)
            continue;
        current.id = top.new_id;
        symbols[next_position].removed = true;
        current.next = symbols[next_position].next;
        if (current.next >= 0)
            symbols[static_cast<std::size_t>(current.next)].previous =
                static_cast<std::int64_t>(top.position);
        if (current.previous >= 0) {
            const auto previous = static_cast<std::size_t>(current.previous);
            if (const auto* left = find_merge(symbols[previous].id, current.id))
                queue.push({left->rank, previous, left->new_id});
        }
        if (current.next >= 0) {
            if (const auto* right =
                    find_merge(current.id, symbols[static_cast<std::size_t>(current.next)].id))
                queue.push({right->rank, top.position, right->new_id});
        }
    }
    for (const auto& symbol : symbols) {
        if (!symbol.removed)
            ids.push_back(symbol.id);
    }
}

void WordTokenizer::encode_piece(std::string_view piece, std::vector<std::int32_t>& ids) const {
    // ByteLevel add_prefix_space prepends one space to every piece that does
    // not already start with one; the regex then keeps it on the first match.
    std::string spaced;
    if (piece.empty() || piece.front() != ' ')
        spaced.push_back(' ');
    spaced.append(piece);
    for (const auto pre_token : pre_tokenize(spaced))
        encode_pre_token(pre_token, ids);
}

std::vector<std::int32_t> WordTokenizer::encode_word(std::string_view word) const {
    const auto points = decode_utf8(word);
    std::vector<std::int32_t> ids;
    std::size_t piece_begin = 0;
    std::size_t index = 0;
    const auto flush = [&](std::size_t piece_end) {
        if (piece_end > piece_begin)
            encode_piece(word.substr(piece_begin, piece_end - piece_begin), ids);
    };
    while (index < points.size()) {
        const AddedToken* match = nullptr;
        const std::size_t at = points[index].begin;
        for (const auto& token : added_) {
            if (word.compare(at, token.content.size(), token.content) == 0 &&
                (match == nullptr || token.content.size() > match->content.size()))
                match = &token;
        }
        if (match == nullptr) {
            ++index;
            continue;
        }
        std::size_t piece_end = at;
        if (match->lstrip) {
            // lstrip folds the White_Space before the token into its match.
            const auto before = decode_utf8(word.substr(piece_begin, at - piece_begin));
            std::size_t kept = before.size();
            while (kept > 0 && is_white_space(before[kept - 1].value))
                --kept;
            piece_end = piece_begin + (kept == 0 ? 0 : before[kept - 1].end);
        }
        flush(piece_end);
        ids.push_back(match->id);
        std::size_t after = at + match->content.size();
        while (index < points.size() && points[index].begin < after)
            ++index;
        if (match->rstrip) {
            while (index < points.size() && is_white_space(points[index].value))
                ++index;
            after = index < points.size() ? points[index].begin : word.size();
        }
        piece_begin = after;
    }
    flush(word.size());
    return ids;
}

} // namespace trtmc::layoutlm
