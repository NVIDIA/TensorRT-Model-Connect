/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <functional>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unicode/regex.h>
#include <unicode/unistr.h>
#include <unordered_map>
#include <unordered_set>

namespace trtmc::nemotron_bolt {
namespace {
std::string utf8(const icu::UnicodeString& text) {
    std::string result;
    text.toUTF8String(result);
    return result;
}
uint64_t pair_key(int32_t a, int32_t b) {
    return (uint64_t(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}
} // namespace

struct Tokenizer::Impl {
    std::unique_ptr<icu::RegexPattern> pattern;
    std::unordered_map<std::string, int32_t> vocab;
    struct Merge {
        int32_t rank, token;
    };
    std::unordered_map<uint64_t, Merge> merges;
    std::unordered_map<int32_t, std::string> inverse;
    std::unordered_set<int32_t> special_ids;
    std::vector<std::pair<std::string, int32_t>> added;
    std::array<std::string, 256> byte_encoder;
    std::unordered_map<UChar32, unsigned char> byte_decoder;
    bool ignore_merges{};

    explicit Impl(std::string_view serialized) {
        const auto j = nlohmann::json::parse(serialized);
        const auto& model = j.at("model");
        const auto& pre = j.at("pre_tokenizer");
        if (model.at("type") != "BPE" || !j.at("normalizer").is_null() ||
            pre.at("type") != "Sequence" || model.value("byte_fallback", false) ||
            !model.at("dropout").is_null())
            throw std::invalid_argument("Unsupported Nemotron Bolt tokenizer contract");
        const auto& parts = pre.at("pretokenizers");
        if (parts.size() != 2 || parts[0].at("type") != "Split" ||
            parts[0].at("behavior") != "Isolated" || parts[0].value("invert", false) ||
            parts[1].at("type") != "ByteLevel" || parts[1].value("add_prefix_space", false) ||
            parts[1].value("use_regex", true) || j.at("decoder").at("type") != "ByteLevel")
            throw std::invalid_argument("Unsupported Nemotron Bolt pre-tokenizer");
        UErrorCode status = U_ZERO_ERROR;
        const auto expression = parts[0].at("pattern").at("Regex").get<std::string>();
        pattern.reset(
            icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(expression), 0, status));
        if (U_FAILURE(status) || !pattern)
            throw std::invalid_argument("Invalid Nemotron Bolt tokenizer split expression");
        vocab = model.at("vocab").get<decltype(vocab)>();
        for (const auto& [token, id] : vocab)
            inverse.emplace(id, token);
        int rank = 0;
        for (const auto& merge : model.at("merges")) {
            std::string a, b;
            if (merge.is_array()) {
                a = merge.at(0);
                b = merge.at(1);
            } else {
                const auto text = merge.get<std::string>();
                const auto space = text.find(' ');
                if (space == std::string::npos)
                    throw std::invalid_argument("Malformed Nemotron Bolt BPE merge");
                a = text.substr(0, space);
                b = text.substr(space + 1);
            }
            merges.emplace(pair_key(vocab.at(a), vocab.at(b)), Merge{rank++, vocab.at(a + b)});
        }
        ignore_merges = model.value("ignore_merges", false);
        for (const auto& item : j.at("added_tokens")) {
            // Matching flags must not be silently discarded.
            if (item.value("single_word", false) || item.value("lstrip", false) ||
                item.value("rstrip", false) || item.value("normalized", false))
                throw std::invalid_argument("Unsupported Nemotron Bolt added-token matching flags");
            std::string text = item.at("content");
            int32_t id = item.at("id");
            if (text.empty())
                throw std::invalid_argument("Empty Nemotron Bolt added token");
            added.emplace_back(text, id);
            inverse[id] = text;
            if (item.value("special", false))
                special_ids.insert(id);
        }
        int extra = 256;
        for (int b = 0; b < 256; ++b) {
            const bool literal = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
            const UChar32 cp = literal ? b : extra++;
            byte_encoder[b] = utf8(icu::UnicodeString(cp));
            byte_decoder.emplace(cp, static_cast<unsigned char>(b));
        }
    }

    void piece(const std::string& bytes, std::vector<int32_t>& result) const {
        if (ignore_merges) {
            std::string encoded;
            for (unsigned char b : bytes)
                encoded += byte_encoder[b];
            const auto found = vocab.find(encoded);
            if (found != vocab.end()) {
                result.push_back(found->second);
                return;
            }
        }
        // Stable positions break equal-rank ties from left to right. Only the
        // two neighbors of a merge need new candidates, even for long words.
        const size_t end = bytes.size();
        struct Symbol {
            int32_t id;
            size_t previous, next;
        };
        std::vector<Symbol> units;
        units.reserve(end);
        for (size_t i = 0; i < end; ++i)
            units.push_back({vocab.at(byte_encoder[static_cast<unsigned char>(bytes[i])]),
                             i == 0 ? end : i - 1, i + 1});
        using Candidate = std::tuple<int32_t, size_t>;
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;
        auto offer = [&](size_t left) {
            if (left == end || units[left].next == end)
                return;
            const auto it = merges.find(pair_key(units[left].id, units[units[left].next].id));
            if (it != merges.end())
                queue.emplace(it->second.rank, left);
        };
        for (size_t i = 0; i < end; ++i)
            offer(i);
        while (!queue.empty()) {
            const auto [rank, left] = queue.top();
            queue.pop();
            const size_t right = units[left].next;
            if (right == end)
                continue;
            const auto it = merges.find(pair_key(units[left].id, units[right].id));
            if (it == merges.end() || it->second.rank != rank)
                continue; // One of the original neighbors already merged.
            units[left].id = it->second.token;
            units[left].next = units[right].next;
            units[right].next = end;
            if (units[left].next != end)
                units[units[left].next].previous = left;
            offer(units[left].previous);
            offer(left);
        }
        for (size_t i = 0; i != end; i = units[i].next)
            result.push_back(units[i].id);
    }

    void ordinary(std::string_view bytes, std::vector<int32_t>& result) const {
        const auto input =
            icu::UnicodeString::fromUTF8(icu::StringPiece(bytes.data(), bytes.size()));
        UErrorCode status = U_ZERO_ERROR;
        std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(input, status));
        if (U_FAILURE(status) || !matcher)
            throw std::runtime_error("Nemotron Bolt regex allocation failed");
        int32_t end = 0;
        while (matcher->find(status)) {
            const int32_t start = matcher->start(status);
            if (start > end)
                piece(utf8(input.tempSubStringBetween(end, start)), result);
            end = matcher->end(status);
            piece(utf8(input.tempSubStringBetween(start, end)), result);
        }
        if (U_FAILURE(status))
            throw std::runtime_error("Nemotron Bolt token split failed");
        if (end < input.length())
            piece(utf8(input.tempSubString(end)), result);
    }
};

Tokenizer::Tokenizer(std::string_view serialized) : impl_(std::make_unique<Impl>(serialized)) {}
Tokenizer::~Tokenizer() = default;
std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<int32_t> result;
    size_t offset = 0;
    // Search an absent or distant added token only once, not once per media marker.
    std::vector<size_t> next(impl_->added.size());
    for (size_t i = 0; i < next.size(); ++i)
        next[i] = text.find(impl_->added[i].first);
    while (offset < text.size()) {
        size_t first = std::string_view::npos, length = 0;
        int32_t id = 0;
        for (size_t i = 0; i < impl_->added.size(); ++i) {
            const auto& [token, token_id] = impl_->added[i];
            if (next[i] != std::string_view::npos && next[i] < offset)
                next[i] = text.find(token, offset);
            const auto at = next[i];
            if (at != std::string_view::npos &&
                (at < first || (at == first && token.size() > length))) {
                first = at;
                length = token.size();
                id = token_id;
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
    std::string result;
    for (int32_t id : ids) {
        if (impl_->special_ids.count(id))
            continue;
        const auto found = impl_->inverse.find(id);
        if (found == impl_->inverse.end())
            throw std::invalid_argument("Invalid Nemotron Bolt token ID");
        const auto token = icu::UnicodeString::fromUTF8(found->second);
        for (int32_t i = 0; i < token.length();) {
            const UChar32 cp = token.char32At(i);
            i += U16_LENGTH(cp);
            const auto byte = impl_->byte_decoder.find(cp);
            if (byte == impl_->byte_decoder.end())
                result += utf8(icu::UnicodeString(cp));
            else
                result.push_back(static_cast<char>(byte->second));
        }
    }
    // Byte BPE can end mid-codepoint. Match replacement decoding rather than emit invalid UTF-8.
    return utf8(icu::UnicodeString::fromUTF8(result));
}
} // namespace trtmc::nemotron_bolt
