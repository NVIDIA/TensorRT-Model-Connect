/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/mobilebert/runtime/tokenizer.h"
#include "families/mobilebert/runtime/unicode_tables.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using mobilebert_unicode::CodeExpansion;
using mobilebert_unicode::CodeRange;
using mobilebert_unicode::CodeShift;
using mobilebert_unicode::kDropped;
using mobilebert_unicode::kExpanded;
using mobilebert_unicode::kIsolated;
using mobilebert_unicode::kPunctuation;
using mobilebert_unicode::kReorderable;
using mobilebert_unicode::kShifted;
using mobilebert_unicode::kSpaced;
using mobilebert_unicode::kWhitespace;

template <std::size_t N>
bool contains(const CodeRange (&ranges)[N], char32_t cp) {
    const auto* found =
        std::lower_bound(std::begin(ranges), std::end(ranges), cp,
                         [](const CodeRange& range, char32_t value) { return range.hi < value; });
    return found != std::end(ranges) && found->lo <= cp;
}

// Refuses text that is not well-formed UTF-8 (truncated or stray bytes, overlong
// forms, surrogates, values past U+10FFFF): the reference only ever sees valid text,
// so there is no reference behaviour to reproduce for it.
void require_utf8(const std::string& text) {
    for (std::size_t pos = 0; pos < text.size();) {
        const auto lead = static_cast<unsigned char>(text[pos]);
        const std::size_t length = lead < 0x80             ? 1
                                   : (lead & 0xE0) == 0xC0 ? 2
                                   : (lead & 0xF0) == 0xE0 ? 3
                                   : (lead & 0xF8) == 0xF0 ? 4
                                                           : 0;
        if (length == 0 || pos + length > text.size())
            throw std::invalid_argument("mobilebert input is not valid UTF-8");
        char32_t cp = length == 1 ? lead : lead & (0xFF >> (length + 1));
        for (std::size_t i = 1; i < length; ++i) {
            const auto next = static_cast<unsigned char>(text[pos + i]);
            if ((next & 0xC0) != 0x80)
                throw std::invalid_argument("mobilebert input is not valid UTF-8");
            cp = (cp << 6) | (next & 0x3F);
        }
        constexpr char32_t kSmallest[] = {0, 0, 0x80, 0x800, 0x10000};
        if (cp < kSmallest[length] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            throw std::invalid_argument("mobilebert input is not valid UTF-8");
        pos += length;
    }
}

// Decodes one code point of text already known to be well-formed.
char32_t next_code_point(const std::string& text, std::size_t& pos) {
    const auto byte = [&](std::size_t at) { return static_cast<unsigned char>(text[at]); };
    const unsigned char lead = byte(pos);
    std::size_t length = lead < 0x80             ? 1
                         : (lead & 0xE0) == 0xC0 ? 2
                         : (lead & 0xF0) == 0xE0 ? 3
                         : (lead & 0xF8) == 0xF0 ? 4
                                                 : 0;
    if (length == 0 || pos + length > text.size()) {
        ++pos;
        return 0xFFFD;
    }
    char32_t cp = length == 1 ? lead : lead & (0xFF >> (length + 1));
    for (std::size_t i = 1; i < length; ++i) {
        if ((byte(pos + i) & 0xC0) != 0x80) {
            ++pos;
            return 0xFFFD;
        }
        cp = (cp << 6) | (byte(pos + i) & 0x3F);
    }
    pos += length;
    return cp;
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// The uncased BertNormalizer of one code point: clean, separate CJK, strip accents
// (NFD without non-spacing marks), lowercase. See unicode_tables.h.
void normalize_code_point(char32_t cp, std::string& out) {
    if (cp >= 0xAC00 && cp <= 0xD7A3) {
        const char32_t index = cp - 0xAC00;
        append_utf8(out, 0x1100 + index / 588);
        append_utf8(out, 0x1161 + (index % 588) / 28);
        if (index % 28)
            append_utf8(out, 0x11A7 + index % 28);
        return;
    }
    if (contains(kDropped, cp))
        return;
    if (contains(kSpaced, cp)) {
        out.push_back(' ');
        return;
    }
    if (contains(kIsolated, cp)) {
        out.push_back(' ');
        append_utf8(out, cp);
        out.push_back(' ');
        return;
    }
    const auto* shift =
        std::lower_bound(std::begin(kShifted), std::end(kShifted), cp,
                         [](const CodeShift& range, char32_t value) { return range.hi < value; });
    if (shift != std::end(kShifted) && shift->lo <= cp) {
        const auto mapped = static_cast<char32_t>(static_cast<std::int64_t>(cp) + shift->delta);
        if (shift->isolated)
            out.push_back(' ');
        append_utf8(out, mapped);
        if (shift->isolated)
            out.push_back(' ');
        return;
    }
    const auto* expansion = std::lower_bound(
        std::begin(kExpanded), std::end(kExpanded), cp,
        [](const CodeExpansion& entry, char32_t value) { return entry.code < value; });
    if (expansion != std::end(kExpanded) && expansion->code == cp) {
        for (std::size_t i = 0; i < expansion->count; ++i)
            append_utf8(out, expansion->out[i]);
        return;
    }
    append_utf8(out, cp);
}

std::string normalize(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t pos = 0; pos < text.size();)
        normalize_code_point(next_code_point(text, pos), out);
    return out;
}

// BertPreTokenizer: whitespace separates words; punctuation is a word of its own.
std::vector<std::string> pre_tokenize(const std::string& text) {
    std::vector<std::string> words;
    std::string current;
    for (std::size_t pos = 0; pos < text.size();) {
        const std::size_t start = pos;
        const char32_t cp = next_code_point(text, pos);
        const bool punctuation = contains(kPunctuation, cp);
        if (punctuation || contains(kWhitespace, cp)) {
            if (!current.empty())
                words.push_back(std::move(current));
            current.clear();
            if (punctuation)
                words.push_back(text.substr(start, pos - start));
            continue;
        }
        current.append(text, start, pos - start);
    }
    if (!current.empty())
        words.push_back(std::move(current));
    return words;
}

std::size_t code_point_count(const std::string& text) {
    std::size_t count = 0;
    for (std::size_t pos = 0; pos < text.size(); ++count)
        next_code_point(text, pos);
    return count;
}

// Start of the last code point of text[start, end).
std::size_t previous_boundary(const std::string& text, std::size_t start, std::size_t end) {
    std::size_t pos = end - 1;
    while (pos > start && (static_cast<unsigned char>(text[pos]) & 0xC0) == 0x80)
        --pos;
    return pos;
}

bool holds_reorderable_mark(const std::string& token) {
    for (std::size_t pos = 0; pos < token.size();) {
        if (contains(kReorderable, next_code_point(token, pos)))
            return true;
    }
    return false;
}

class MobileBertTokenizer final : public IMobileBertTokenizer {
  public:
    explicit MobileBertTokenizer(const nlohmann::json& json) {
        read_model(json);
        read_normalizer(json);
        read_added_tokens(json);
        const auto unknown = token_to_id_.find(unk_token_);
        if (unknown == token_to_id_.end())
            throw std::runtime_error("mobilebert tokenizer does not define its unknown token");
        unk_id_ = unknown->second;
    }

    std::vector<std::int32_t> encode(std::string_view input) const override {
        // Added tokens are found in the raw text first, leftmost and then longest,
        // and the text between them is normalised and split on its own.
        const std::string text(input);
        require_utf8(text);
        std::vector<std::int32_t> ids;
        std::size_t start = 0;
        while (start < text.size()) {
            std::size_t next = std::string::npos;
            std::int32_t token_id = -1;
            std::size_t token_size = 0;
            for (const auto& [token, id] : added_tokens_) {
                const auto found = text.find(token, start);
                if (found < next || (found == next && token.size() > token_size)) {
                    next = found;
                    token_id = id;
                    token_size = token.size();
                }
            }
            const auto end = next == std::string::npos ? text.size() : next;
            for (const auto& word : pre_tokenize(normalize(text.substr(start, end - start))))
                encode_word(word, ids);
            if (next == std::string::npos)
                break;
            ids.push_back(token_id);
            start = next + token_size;
        }
        return ids;
    }

    std::size_t size() const override { return id_to_token_.size(); }

    std::int32_t id_for_token(std::string_view token) const override {
        const auto found = token_to_id_.find(std::string(token));
        return found == token_to_id_.end() ? -1 : found->second;
    }

  private:
    // Greedy longest-match-first WordPiece; a word with any piece missing from the
    // vocabulary becomes a single unknown token.
    void encode_word(const std::string& word, std::vector<std::int32_t>& ids) const {
        if (code_point_count(word) > max_chars_per_word_) {
            ids.push_back(unk_id_);
            return;
        }
        std::vector<std::int32_t> pieces;
        std::size_t start = 0;
        while (start < word.size()) {
            std::size_t end = word.size();
            std::int32_t piece = -1;
            while (end > start) {
                const auto candidate =
                    (start > 0 ? prefix_ : std::string()) + word.substr(start, end - start);
                const auto found = vocab_.find(candidate);
                if (found != vocab_.end()) {
                    piece = found->second;
                    break;
                }
                end = previous_boundary(word, start, end);
            }
            if (piece < 0) {
                ids.push_back(unk_id_);
                return;
            }
            pieces.push_back(piece);
            start = end;
        }
        ids.insert(ids.end(), pieces.begin(), pieces.end());
    }

    void read_model(const nlohmann::json& json) {
        const auto& model = json.at("model");
        // Older tokenizer.json files, this checkpoint's among them, carry no model
        // "type"; the reference then recognises WordPiece by its fields.
        const auto type = model.find("type");
        const bool word_piece = type == model.end()
                                    ? model.contains("continuing_subword_prefix") &&
                                          model.contains("max_input_chars_per_word") &&
                                          !model.contains("merges")
                                    : *type == "WordPiece";
        if (!word_piece)
            throw std::runtime_error("mobilebert tokenizer.json is not a WordPiece model");
        const auto pre_tokenizer = json.find("pre_tokenizer");
        if (pre_tokenizer == json.end() || !pre_tokenizer->is_object() ||
            pre_tokenizer->value("type", "") != "BertPreTokenizer")
            throw std::runtime_error("mobilebert tokenizer.json does not use BertPreTokenizer");
        unk_token_ = model.value("unk_token", "[UNK]");
        prefix_ = model.value("continuing_subword_prefix", "##");
        max_chars_per_word_ = model.value("max_input_chars_per_word", std::size_t{100});
        const auto& vocab = model.at("vocab");
        id_to_token_.resize(vocab.size());
        for (const auto& [token, value] : vocab.items()) {
            require_utf8(token);
            const auto id = value.get<std::int32_t>();
            if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size() ||
                !id_to_token_[static_cast<std::size_t>(id)].empty())
                throw std::runtime_error("mobilebert tokenizer vocabulary ids are not 0..n-1");
            // The tables cannot reproduce the reference's reordering of combining marks.
            // That reordering never changes ids while no vocabulary token holds such a
            // mark, so a vocabulary that holds one is refused rather than approximated.
            if (holds_reorderable_mark(token))
                throw std::runtime_error("mobilebert tokenizer vocabulary token '" + token +
                                         "' holds a combining mark this tokenizer cannot order");
            id_to_token_[static_cast<std::size_t>(id)] = token;
            vocab_[token] = id;
            token_to_id_[token] = id;
        }
    }

    // The tables reproduce one normaliser: the uncased BertNormalizer.
    static void read_normalizer(const nlohmann::json& json) {
        const auto& normalizer = json.at("normalizer");
        const auto flag = [&](const char* key) {
            const auto found = normalizer.find(key);
            // strip_accents null follows lowercase, which is required to be true.
            return found == normalizer.end() || found->is_null() ||
                   (found->is_boolean() && found->get<bool>());
        };
        if (!normalizer.is_object() || normalizer.value("type", "") != "BertNormalizer" ||
            !flag("clean_text") || !flag("handle_chinese_chars") || !flag("lowercase") ||
            !flag("strip_accents"))
            throw std::runtime_error(
                "mobilebert tokenizer.json must use the uncased BertNormalizer (clean text, "
                "separate CJK, strip accents, lowercase)");
    }

    void read_added_tokens(const nlohmann::json& json) {
        const auto added = json.find("added_tokens");
        if (added == json.end())
            return;
        for (const auto& entry : *added) {
            const auto content = entry.at("content").get<std::string>();
            const auto id = entry.at("id").get<std::int32_t>();
            if (content.empty() || id < 0)
                throw std::runtime_error("mobilebert tokenizer has an invalid added token");
            if (entry.value("normalized", false) || entry.value("single_word", false) ||
                entry.value("lstrip", false) || entry.value("rstrip", false))
                throw std::runtime_error("mobilebert tokenizer added token '" + content +
                                         "' uses matching options this tokenizer does not "
                                         "implement");
            if (static_cast<std::size_t>(id) >= id_to_token_.size())
                id_to_token_.resize(static_cast<std::size_t>(id) + 1);
            id_to_token_[static_cast<std::size_t>(id)] = content;
            token_to_id_[content] = id;
            added_tokens_.emplace_back(content, id);
        }
    }

    std::vector<std::string> id_to_token_;
    // WordPiece reads the model vocabulary only; added tokens are matched in the
    // raw text and never produced by a word split.
    std::unordered_map<std::string, std::int32_t> vocab_;
    std::unordered_map<std::string, std::int32_t> token_to_id_;
    std::vector<std::pair<std::string, std::int32_t>> added_tokens_;
    std::string unk_token_{"[UNK]"};
    std::string prefix_{"##"};
    std::size_t max_chars_per_word_{100};
    std::int32_t unk_id_{-1};
};

} // namespace

std::unique_ptr<IMobileBertTokenizer> CreateMobileBertTokenizer(const char* tokenizer_json,
                                                                std::size_t size) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(tokenizer_json, tokenizer_json + size);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("mobilebert tokenizer.json is not JSON: ") +
                                 error.what());
    }
    return std::make_unique<MobileBertTokenizer>(json);
}

} // namespace trtmc
