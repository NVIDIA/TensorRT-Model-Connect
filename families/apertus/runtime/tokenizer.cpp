/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/apertus/runtime/tokenizer.h"

#include <array>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unicode/regex.h>
#include <unicode/unistr.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace trtmc::apertus {
namespace {

std::string utf8(const icu::UnicodeString& text) {
    std::string result;
    text.toUTF8String(result);
    return result;
}

std::string pair_key(const std::string& left, const std::string& right) {
    return left + '\0' + right;
}

bool is_null_or_absent(const nlohmann::json& object, const char* name) {
    return !object.contains(name) || object.at(name).is_null();
}

void require_contract(const nlohmann::json& tokenizer) {
    const auto& model = tokenizer.at("model");
    if (model.at("type") != "BPE" || !is_null_or_absent(tokenizer, "normalizer") ||
        !is_null_or_absent(model, "dropout") || model.value("byte_fallback", false) ||
        !is_null_or_absent(model, "continuing_subword_prefix") ||
        !is_null_or_absent(model, "end_of_word_suffix")) {
        throw std::invalid_argument("unsupported Apertus tokenizer model");
    }
    const auto& pre = tokenizer.at("pre_tokenizer");
    if (pre.at("type") != "Sequence")
        throw std::invalid_argument("unsupported Apertus pre-tokenizer");
    const auto& parts = pre.at("pretokenizers");
    if (parts.size() != 2 || parts[0].at("type") != "Split" ||
        parts[0].at("behavior") != "Isolated" || parts[0].value("invert", false) ||
        !parts[0].at("pattern").contains("Regex") || parts[1].at("type") != "ByteLevel" ||
        parts[1].value("add_prefix_space", false) || parts[1].value("use_regex", true) ||
        tokenizer.at("decoder").at("type") != "ByteLevel") {
        throw std::invalid_argument("unsupported Apertus pre-tokenizer");
    }
}

} // namespace

struct Tokenizer::Impl {
    std::unique_ptr<icu::RegexPattern> pattern;
    std::unordered_map<std::string, std::int32_t> vocab;
    std::unordered_map<std::string, std::int32_t> ranks;
    std::unordered_map<std::int32_t, std::string> inverse;
    std::unordered_set<std::int32_t> special_ids;
    std::vector<std::pair<std::string, std::int32_t>> added;
    std::array<std::string, 256> byte_encoder;
    std::unordered_map<UChar32, unsigned char> byte_decoder;
    bool ignore_merges{false};
    std::int32_t id_count{0};

    explicit Impl(std::string_view serialized) {
        const auto tokenizer = nlohmann::json::parse(serialized);
        require_contract(tokenizer);
        const auto& model = tokenizer.at("model");
        const auto expression = tokenizer.at("pre_tokenizer")
                                    .at("pretokenizers")[0]
                                    .at("pattern")
                                    .at("Regex")
                                    .get<std::string>();
        UErrorCode status = U_ZERO_ERROR;
        pattern.reset(
            icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(expression), 0, status));
        if (U_FAILURE(status) || !pattern)
            throw std::invalid_argument("invalid Apertus tokenizer split expression");
        vocab = model.at("vocab").get<decltype(vocab)>();
        for (const auto& [token, id] : vocab) {
            if (id < 0)
                throw std::invalid_argument("negative Apertus token ID");
            inverse.emplace(id, token);
            id_count = std::max(id_count, id + 1);
        }
        std::int32_t rank = 0;
        for (const auto& merge : model.at("merges")) {
            std::string left, right;
            if (merge.is_array() && merge.size() == 2) {
                left = merge[0].get<std::string>();
                right = merge[1].get<std::string>();
            } else {
                const auto text = merge.get<std::string>();
                const auto space = text.find(' ');
                if (space == std::string::npos)
                    throw std::invalid_argument("malformed Apertus BPE merge");
                left = text.substr(0, space);
                right = text.substr(space + 1);
            }
            ranks.emplace(pair_key(left, right), rank++);
        }
        ignore_merges = model.value("ignore_merges", false);
        for (const auto& item : tokenizer.at("added_tokens")) {
            // HF matching flags change where an added token may match; none is implemented.
            if (item.value("single_word", false) || item.value("lstrip", false) ||
                item.value("rstrip", false) || item.value("normalized", false))
                throw std::invalid_argument("unsupported Apertus added-token matching flags");
            auto text = item.at("content").get<std::string>();
            const auto id = item.at("id").get<std::int32_t>();
            if (text.empty() || id < 0)
                throw std::invalid_argument("invalid Apertus added token");
            inverse[id] = text;
            id_count = std::max(id_count, id + 1);
            if (item.value("special", false))
                special_ids.insert(id);
            added.emplace_back(std::move(text), id);
        }
        // GPT-2 byte-to-unicode table: printable bytes map to themselves.
        int extra = 256;
        for (int value = 0; value < 256; ++value) {
            const bool printable =
                (value >= 33 && value <= 126) || (value >= 161 && value <= 172) || value >= 174;
            const UChar32 cp = printable ? value : extra++;
            byte_encoder[static_cast<std::size_t>(value)] = utf8(icu::UnicodeString(cp));
            byte_decoder.emplace(cp, static_cast<unsigned char>(value));
        }
    }

    void piece(const std::string& bytes, std::vector<std::int32_t>& result) const {
        std::vector<std::string> units;
        std::string encoded;
        for (const unsigned char value : bytes) {
            units.push_back(byte_encoder[value]);
            encoded += byte_encoder[value];
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
            std::size_t best = units.size();
            std::int32_t best_rank = std::numeric_limits<std::int32_t>::max();
            for (std::size_t index = 0; index + 1 < units.size(); ++index) {
                const auto found = ranks.find(pair_key(units[index], units[index + 1]));
                if (found != ranks.end() && found->second < best_rank) {
                    best_rank = found->second;
                    best = index;
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
                throw std::invalid_argument("Apertus vocabulary is missing a byte-level unit");
            result.push_back(found->second);
        }
    }

    void ordinary(std::string_view bytes, std::vector<std::int32_t>& result) const {
        if (bytes.empty())
            return;
        const auto input = icu::UnicodeString::fromUTF8(
            icu::StringPiece(bytes.data(), static_cast<std::int32_t>(bytes.size())));
        UErrorCode status = U_ZERO_ERROR;
        std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(input, status));
        if (U_FAILURE(status) || !matcher)
            throw std::runtime_error("Apertus tokenizer regex allocation failed");
        std::int32_t end = 0;
        while (matcher->find(status)) {
            const std::int32_t start = matcher->start(status);
            if (start > end)
                piece(utf8(input.tempSubStringBetween(end, start)), result);
            end = matcher->end(status);
            piece(utf8(input.tempSubStringBetween(start, end)), result);
        }
        if (U_FAILURE(status))
            throw std::runtime_error("Apertus tokenizer split failed");
        if (end < input.length())
            piece(utf8(input.tempSubString(end)), result);
    }
};

Tokenizer::Tokenizer(std::string_view serialized) : impl_(std::make_unique<Impl>(serialized)) {}
Tokenizer::~Tokenizer() = default;

std::int32_t Tokenizer::vocab_size() const {
    return impl_->id_count;
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<std::int32_t> result;
    std::size_t offset = 0;
    // Next occurrence of each added token, searched again only after it is passed.
    std::vector<std::size_t> next(impl_->added.size());
    for (std::size_t index = 0; index < next.size(); ++index)
        next[index] = text.find(impl_->added[index].first);
    while (offset < text.size()) {
        std::size_t first = std::string_view::npos;
        std::size_t length = 0;
        std::int32_t id = 0;
        for (std::size_t index = 0; index < impl_->added.size(); ++index) {
            const auto& [token, token_id] = impl_->added[index];
            if (next[index] != std::string_view::npos && next[index] < offset)
                next[index] = text.find(token, offset);
            const auto at = next[index];
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

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids) const {
    std::string bytes;
    for (const auto id : ids) {
        if (impl_->special_ids.count(id) != 0)
            continue;
        const auto found = impl_->inverse.find(id);
        if (found == impl_->inverse.end())
            throw std::invalid_argument("invalid Apertus token ID");
        const auto token = icu::UnicodeString::fromUTF8(found->second);
        for (std::int32_t index = 0; index < token.length();) {
            const UChar32 cp = token.char32At(index);
            index += U16_LENGTH(cp);
            const auto byte = impl_->byte_decoder.find(cp);
            if (byte == impl_->byte_decoder.end())
                bytes += utf8(icu::UnicodeString(cp));
            else
                bytes.push_back(static_cast<char>(byte->second));
        }
    }
    // Byte-level BPE can stop inside a code point; decode like a lossy UTF-8 reader.
    return utf8(icu::UnicodeString::fromUTF8(bytes));
}

} // namespace trtmc::apertus
