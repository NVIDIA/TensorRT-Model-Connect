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
#include <unicode/unistr.h>
#include <unordered_map>
#include <unordered_set>

namespace trtmc::hunyuan_vl {
namespace {
std::string utf8(const icu::UnicodeString& text) {
    std::string result;
    text.toUTF8String(result);
    return result;
}
std::string pair_key(const std::string& a, const std::string& b) {
    return a + '\0' + b;
}
// The serialized expressions target Oniguruma, whose Unicode \s is the White_Space property.
// ICU's \s omits U+000B and U+0085, so spell the property out.
std::string onig_whitespace(const std::string& expression) {
    std::string result;
    for (size_t i = 0; i < expression.size(); ++i) {
        if (expression[i] != '\\' || i + 1 == expression.size()) {
            result += expression[i];
            continue;
        }
        const char next = expression[++i];
        if (next == 's')
            result += "\\p{White_Space}";
        else if (next == 'S')
            result += "\\P{White_Space}";
        else {
            result += '\\';
            result += next;
        }
    }
    return result;
}
} // namespace

struct Tokenizer::Impl {
    std::vector<std::unique_ptr<icu::RegexPattern>> splits;
    std::unordered_map<std::string, int32_t> vocab, ranks;
    std::unordered_map<int32_t, std::string> inverse;
    std::unordered_set<int32_t> special_ids;
    std::vector<std::pair<std::string, int32_t>> added;
    std::array<std::string, 256> byte_encoder;
    std::unordered_map<UChar32, unsigned char> byte_decoder;
    bool ignore_merges{};

    explicit Impl(std::string_view serialized) {
        const auto j = nlohmann::json::parse(serialized);
        const auto& model = j.at("model");
        const auto& normalizer = j.at("normalizer");
        if (model.at("type") != "BPE" || model.value("byte_fallback", false) ||
            !model.at("dropout").is_null() ||
            !model.value("continuing_subword_prefix", nlohmann::json()).is_null() ||
            !model.value("end_of_word_suffix", nlohmann::json()).is_null() ||
            !(normalizer.is_null() ||
              (normalizer.at("type") == "Sequence" && normalizer.at("normalizers").empty())))
            throw std::invalid_argument("Unsupported HunYuanVL tokenizer model");
        const auto& pre = j.at("pre_tokenizer");
        if (pre.at("type") != "Sequence" || pre.at("pretokenizers").size() < 2)
            throw std::invalid_argument("Unsupported HunYuanVL pre-tokenizer");
        const auto& parts = pre.at("pretokenizers");
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            const auto& part = parts[i];
            if (part.at("type") != "Split" || part.at("behavior") != "Isolated" ||
                part.value("invert", false) || !part.at("pattern").contains("Regex"))
                throw std::invalid_argument("Unsupported HunYuanVL split pre-tokenizer");
            UErrorCode status = U_ZERO_ERROR;
            const auto expression = onig_whitespace(part.at("pattern").at("Regex"));
            splits.emplace_back(
                icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(expression), 0, status));
            if (U_FAILURE(status) || !splits.back())
                throw std::invalid_argument("Invalid HunYuanVL tokenizer split expression");
        }
        const auto& last = parts.back();
        if (last.at("type") != "ByteLevel" || last.value("add_prefix_space", false) ||
            last.value("use_regex", true) || j.at("decoder").at("type") != "ByteLevel")
            throw std::invalid_argument("Unsupported HunYuanVL byte-level stage");
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
                    throw std::invalid_argument("Malformed HunYuanVL BPE merge");
                a = text.substr(0, space);
                b = text.substr(space + 1);
            }
            ranks.emplace(pair_key(a, b), rank++);
        }
        ignore_merges = model.value("ignore_merges", false);
        for (const auto& item : j.at("added_tokens")) {
            // Matching flags must not be silently discarded.
            if (item.value("single_word", false) || item.value("lstrip", false) ||
                item.value("rstrip", false) || item.value("normalized", false))
                throw std::invalid_argument("Unsupported HunYuanVL added-token matching flags");
            std::string text = item.at("content");
            const int32_t id = item.at("id");
            if (text.empty())
                throw std::invalid_argument("Empty HunYuanVL added token");
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
        std::vector<std::string> units;
        std::string encoded;
        for (unsigned char b : bytes) {
            encoded += byte_encoder[b];
            // Without an unknown token or byte fallback, the reference drops characters the
            // vocabulary lacks (this vocabulary has no carriage return) before merging.
            if (vocab.count(byte_encoder[b]))
                units.push_back(byte_encoder[b]);
        }
        if (ignore_merges) {
            const auto found = vocab.find(encoded);
            if (found != vocab.end()) {
                result.push_back(found->second);
                return;
            }
        }
        // Merge the lowest-ranked adjacent pair, leftmost first, until none applies.
        while (units.size() > 1) {
            size_t best = units.size();
            int rank = std::numeric_limits<int>::max();
            for (size_t i = 0; i + 1 < units.size(); ++i) {
                const auto found = ranks.find(pair_key(units[i], units[i + 1]));
                if (found != ranks.end() && found->second < rank) {
                    rank = found->second;
                    best = i;
                }
            }
            if (best == units.size())
                break;
            units[best] += units[best + 1];
            units.erase(units.begin() + static_cast<std::ptrdiff_t>(best) + 1);
        }
        for (const auto& token : units) {
            const auto found = vocab.find(token);
            if (found == vocab.end())
                throw std::invalid_argument(
                    "HunYuanVL merge produced a token outside the vocabulary");
            result.push_back(found->second);
        }
    }

    // Isolate every match of one split stage; the text between matches stays a piece too.
    void split(const icu::RegexPattern& pattern, const icu::UnicodeString& input,
               std::vector<icu::UnicodeString>& output) const {
        UErrorCode status = U_ZERO_ERROR;
        std::unique_ptr<icu::RegexMatcher> matcher(pattern.matcher(input, status));
        if (U_FAILURE(status) || !matcher)
            throw std::runtime_error("HunYuanVL regex allocation failed");
        // Pieces outlive this stage's input, so copy them (temporary substrings alias it).
        const auto keep = [&](int32_t start, int32_t stop) {
            output.emplace_back();
            input.extractBetween(start, stop, output.back());
        };
        int32_t end = 0;
        while (matcher->find(status)) {
            const int32_t start = matcher->start(status);
            const int32_t stop = matcher->end(status);
            if (start > end)
                keep(end, start);
            if (stop > start)
                keep(start, stop);
            end = stop;
        }
        if (U_FAILURE(status))
            throw std::runtime_error("HunYuanVL token split failed");
        if (end < input.length())
            keep(end, input.length());
    }

    void ordinary(std::string_view bytes, std::vector<int32_t>& result) const {
        if (bytes.empty())
            return;
        std::vector<icu::UnicodeString> pieces{
            icu::UnicodeString::fromUTF8(icu::StringPiece(bytes.data(), int32_t(bytes.size())))};
        for (const auto& pattern : splits) {
            std::vector<icu::UnicodeString> next;
            for (const auto& text : pieces)
                split(*pattern, text, next);
            pieces.swap(next);
        }
        for (const auto& text : pieces)
            piece(utf8(text), result);
    }
};

Tokenizer::Tokenizer(std::string_view serialized) : impl_(std::make_unique<Impl>(serialized)) {}
Tokenizer::~Tokenizer() = default;

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<int32_t> result;
    size_t offset = 0;
    // Search an absent or distant added token only once, not once per occurrence.
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
    std::string bytes;
    for (const int32_t id : ids) {
        if (impl_->special_ids.count(id))
            continue;
        const auto found = impl_->inverse.find(id);
        if (found == impl_->inverse.end())
            throw std::invalid_argument("Invalid HunYuanVL token ID");
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
    // Byte BPE can end mid-codepoint: decode with replacement, never emit invalid UTF-8.
    return utf8(icu::UnicodeString::fromUTF8(bytes));
}
} // namespace trtmc::hunyuan_vl
