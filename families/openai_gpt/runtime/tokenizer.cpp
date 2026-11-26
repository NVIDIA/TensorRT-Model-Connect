/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/openai_gpt/runtime/tokenizer.h"

#include "families/openai_gpt/runtime/unicode_tables.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc::openai_gpt {
namespace {

using openai_gpt_unicode::CodeExpansion;
using openai_gpt_unicode::CodeRange;
using openai_gpt_unicode::CodeShift;
using openai_gpt_unicode::kDropped;
using openai_gpt_unicode::kExpanded;
using openai_gpt_unicode::kIsolated;
using openai_gpt_unicode::kPunctuation;
using openai_gpt_unicode::kReorderable;
using openai_gpt_unicode::kShifted;
using openai_gpt_unicode::kSpaced;
using openai_gpt_unicode::kWhitespace;

template <std::size_t N>
bool contains(const CodeRange (&ranges)[N], char32_t cp) {
    const auto* found =
        std::lower_bound(std::begin(ranges), std::end(ranges), cp,
                         [](const CodeRange& range, char32_t value) { return range.hi < value; });
    return found != std::end(ranges) && found->lo <= cp;
}

// Decodes the code point at pos and moves past it. Text that is not well-formed UTF-8
// (truncated or stray bytes, overlong forms, surrogates, values past U+10FFFF) is refused:
// the reference only ever sees valid text, so there is no reference behaviour for it.
char32_t next_code_point(std::string_view text, std::size_t& pos) {
    const auto lead = static_cast<unsigned char>(text[pos]);
    const std::size_t length = lead < 0x80             ? 1
                               : (lead & 0xE0) == 0xC0 ? 2
                               : (lead & 0xF0) == 0xE0 ? 3
                               : (lead & 0xF8) == 0xF0 ? 4
                                                       : 0;
    if (length == 0 || length > text.size() - pos)
        throw std::invalid_argument("OpenAI GPT text is not valid UTF-8");
    char32_t cp = length == 1 ? lead : lead & (0xFF >> (length + 1));
    for (std::size_t i = 1; i < length; ++i) {
        const auto next = static_cast<unsigned char>(text[pos + i]);
        if ((next & 0xC0) != 0x80)
            throw std::invalid_argument("OpenAI GPT text is not valid UTF-8");
        cp = (cp << 6) | (next & 0x3F);
    }
    constexpr char32_t kSmallest[] = {0, 0, 0x80, 0x800, 0x10000};
    if (cp < kSmallest[length] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
        throw std::invalid_argument("OpenAI GPT text is not valid UTF-8");
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

// Code points of a token; throws unless the token is well-formed UTF-8. Reports whether
// the token holds a combining mark that canonical reordering could move.
std::size_t token_code_points(const std::string& token, bool& reorderable) {
    std::size_t count = 0;
    for (std::size_t pos = 0; pos < token.size(); ++count) {
        if (contains(kReorderable, next_code_point(token, pos)))
            reorderable = true;
    }
    return count;
}

std::runtime_error contract_error(const std::string& message) {
    return std::runtime_error("OpenAI GPT tokenizer.json " + message);
}

// A boolean option; absent selects `fallback`, any other type is refused.
bool flag(const nlohmann::json& section, const char* key, bool fallback) {
    const auto found = section.find(key);
    if (found == section.end())
        return fallback;
    if (!found->is_boolean())
        throw contract_error(std::string("field '") + key + "' must be a boolean");
    return found->get<bool>();
}

// A boolean the reference requires to be present.
bool required_flag(const nlohmann::json& section, const char* key) {
    if (!section.contains(key))
        throw contract_error(std::string("field '") + key + "' is missing");
    return flag(section, key, false);
}

bool is_null_or_absent(const nlohmann::json& section, const char* key) {
    const auto found = section.find(key);
    return found == section.end() || found->is_null();
}

std::string text_field(const nlohmann::json& section, const char* key) {
    const auto found = section.find(key);
    if (found == section.end() || !found->is_string())
        throw contract_error(std::string("field '") + key + "' must be a string");
    return found->get<std::string>();
}

std::uint64_t pair_key(std::int32_t left, std::int32_t right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
           static_cast<std::uint32_t>(right);
}

struct MergeRule {
    std::int32_t rank;
    std::int32_t merged;
};

struct Symbol {
    std::int32_t id;
    std::int32_t prev;
    std::int32_t next;
    bool removed;
};

struct Candidate {
    std::int32_t rank;
    std::int32_t pos;
    std::int32_t merged;
};

// Orders the queue as the reference does: lowest rank first, then leftmost position.
struct LaterCandidate {
    bool operator()(const Candidate& a, const Candidate& b) const {
        return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;
    }
};

std::invalid_argument too_many_tokens(std::size_t max_tokens) {
    return std::invalid_argument("OpenAI GPT text has more than " + std::to_string(max_tokens) +
                                 " tokens");
}

} // namespace

struct Tokenizer::Impl {
    std::unordered_map<std::string, std::int32_t> vocab;
    std::vector<std::string> tokens;
    std::vector<bool> special;
    std::vector<std::pair<std::string, std::int32_t>> added;
    std::unordered_map<std::uint64_t, MergeRule> merges;
    std::string suffix;
    std::int32_t unk_id{-1};
    // Most input characters one token can cover: a word of n characters has at least
    // ceil(n / widest) tokens.
    std::size_t widest{1};

    void parse(const nlohmann::json& json);
    void read_model(const nlohmann::json& model);
    void read_merges(const nlohmann::json& merges_json);
    void read_added_tokens(const nlohmann::json& json);

    // The longest added token that starts at pos, or nullptr.
    const std::pair<std::string, std::int32_t>* added_at(std::string_view text,
                                                         std::size_t pos) const {
        const std::pair<std::string, std::int32_t>* found = nullptr;
        for (const auto& entry : added) {
            if ((found == nullptr || entry.first.size() > found->first.size()) &&
                text.substr(pos, entry.first.size()) == entry.first)
                found = &entry;
        }
        return found;
    }

    // Whether a word of `characters` characters must have more than `budget` tokens.
    bool exceeds(std::size_t characters, std::size_t budget) const {
        return budget < std::numeric_limits<std::size_t>::max() / widest &&
               characters > budget * widest;
    }

    void encode_word(const std::string& word, const std::vector<std::size_t>& starts,
                     std::vector<std::int32_t>& out, std::size_t max_tokens) const;
};

void Tokenizer::Impl::parse(const nlohmann::json& json) {
    if (!json.is_object())
        throw contract_error("must contain one JSON object");
    if (!is_null_or_absent(json, "truncation") || !is_null_or_absent(json, "padding"))
        throw contract_error("must neither truncate nor pad");
    const auto normalizer = json.find("normalizer");
    if (normalizer == json.end() || !normalizer->is_object() ||
        normalizer->value("type", "") != "BertNormalizer" ||
        !required_flag(*normalizer, "clean_text") ||
        !required_flag(*normalizer, "handle_chinese_chars") ||
        !required_flag(*normalizer, "lowercase") ||
        !(is_null_or_absent(*normalizer, "strip_accents") ||
          flag(*normalizer, "strip_accents", true)))
        throw contract_error("must use the uncased BertNormalizer (clean text, separate CJK, "
                             "strip accents, lowercase)");
    const auto pre_tokenizer = json.find("pre_tokenizer");
    if (pre_tokenizer == json.end() || !pre_tokenizer->is_object() ||
        pre_tokenizer->value("type", "") != "BertPreTokenizer")
        throw contract_error("must use BertPreTokenizer");
    if (!is_null_or_absent(json, "post_processor"))
        throw contract_error("must not add special tokens (no post-processor)");
    const auto model = json.find("model");
    if (model == json.end())
        throw contract_error("has no model");
    read_model(*model);
    const auto decoder = json.find("decoder");
    if (decoder == json.end() || !decoder->is_object() ||
        decoder->value("type", "") != "BPEDecoder" || text_field(*decoder, "suffix") != suffix)
        throw contract_error("must decode with BPEDecoder of the model suffix");
    read_added_tokens(json);
}

void Tokenizer::Impl::read_model(const nlohmann::json& model) {
    if (!model.is_object() || model.value("type", "BPE") != "BPE" || !model.contains("merges"))
        throw contract_error("must hold a BPE model");
    if (!is_null_or_absent(model, "dropout"))
        throw contract_error("BPE dropout is not supported");
    if (!is_null_or_absent(model, "continuing_subword_prefix"))
        throw contract_error("BPE continuing_subword_prefix is not supported");
    if (flag(model, "fuse_unk", false) || flag(model, "byte_fallback", false))
        throw contract_error("must map each unknown character to its own token");
    if (flag(model, "ignore_merges", false))
        throw contract_error("BPE ignore_merges is not supported");
    suffix = text_field(model, "end_of_word_suffix");
    if (suffix.empty())
        throw contract_error("BPE must declare an end_of_word_suffix");
    const auto& entries = model.at("vocab");
    if (!entries.is_object() || entries.empty() ||
        entries.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw contract_error("BPE vocabulary must be a non-empty object");
    tokens.assign(entries.size(), std::string());
    std::vector<bool> seen(entries.size(), false);
    for (const auto& [token, value] : entries.items()) {
        // Ids are 0..n-1, each once, so every id names exactly one token.
        if (!value.is_number_integer() || value.get<std::int64_t>() < 0 ||
            value.get<std::int64_t>() >= static_cast<std::int64_t>(entries.size()) ||
            seen[static_cast<std::size_t>(value.get<std::int64_t>())])
            throw contract_error("vocabulary ids must be 0..n-1, each once");
        const auto id = static_cast<std::int32_t>(value.get<std::int64_t>());
        bool reorderable = false;
        const auto length = token.empty() ? 0 : token_code_points(token, reorderable);
        if (length == 0)
            throw contract_error("vocabulary holds an empty token");
        // The tables cannot reproduce the reference's reordering of combining marks. It
        // never changes ids while no vocabulary token holds such a mark (BPE maps each of
        // them to the unknown token), so a vocabulary that holds one is refused.
        if (reorderable)
            throw contract_error("vocabulary token '" + token +
                                 "' holds a combining mark this tokenizer cannot order");
        seen[static_cast<std::size_t>(id)] = true;
        tokens[static_cast<std::size_t>(id)] = token;
        vocab.emplace(token, id);
        widest = std::max(widest, length);
    }
    const auto unknown = vocab.find(text_field(model, "unk_token"));
    if (unknown == vocab.end())
        throw contract_error("unk_token must be a vocabulary token");
    unk_id = unknown->second;
    read_merges(model.at("merges"));
}

void Tokenizer::Impl::read_merges(const nlohmann::json& merges_json) {
    if (!merges_json.is_array() ||
        merges_json.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw contract_error("BPE merges must be a list");
    std::int32_t rank = 0;
    for (const auto& entry : merges_json) {
        std::string left;
        std::string right;
        if (entry.is_string()) {
            const auto text = entry.get<std::string>();
            const auto space = text.find(' ');
            if (space != std::string::npos && text.find(' ', space + 1) == std::string::npos) {
                left = text.substr(0, space);
                right = text.substr(space + 1);
            }
        } else if (entry.is_array() && entry.size() == 2 && entry[0].is_string() &&
                   entry[1].is_string()) {
            left = entry[0].get<std::string>();
            right = entry[1].get<std::string>();
        }
        if (left.empty() || right.empty())
            throw contract_error("BPE merge " + entry.dump() + " is not one pair of tokens");
        const auto a = vocab.find(left);
        const auto b = vocab.find(right);
        const auto merged = vocab.find(left + right);
        if (a == vocab.end() || b == vocab.end() || merged == vocab.end())
            throw contract_error("BPE merge " + entry.dump() + " leaves the vocabulary");
        if (!merges.emplace(pair_key(a->second, b->second), MergeRule{rank, merged->second}).second)
            throw contract_error("BPE merge " + entry.dump() + " is listed twice");
        ++rank;
    }
}

void Tokenizer::Impl::read_added_tokens(const nlohmann::json& json) {
    special.assign(tokens.size(), false);
    const auto entries = json.find("added_tokens");
    if (entries == json.end())
        return;
    if (!entries->is_array())
        throw contract_error("added_tokens must be a list");
    std::vector<bool> listed(tokens.size(), false);
    for (const auto& entry : *entries) {
        if (!entry.is_object())
            throw contract_error("added token must be an object");
        const auto content = text_field(entry, "content");
        const auto id = entry.find("id");
        const auto in_vocab = vocab.find(content);
        // As the builder requires, an added token is a vocabulary token with its own id.
        if (content.empty() || id == entry.end() || !id->is_number_integer() ||
            in_vocab == vocab.end() || id->get<std::int64_t>() != in_vocab->second)
            throw contract_error("added token '" + content +
                                 "' must be a vocabulary token with its vocabulary id");
        // A second entry for one token could change whether decoding skips it.
        if (listed[static_cast<std::size_t>(in_vocab->second)])
            throw contract_error("added token '" + content + "' is listed twice");
        listed[static_cast<std::size_t>(in_vocab->second)] = true;
        if (required_flag(entry, "normalized") || required_flag(entry, "single_word") ||
            required_flag(entry, "lstrip") || required_flag(entry, "rstrip"))
            throw contract_error("added token '" + content +
                                 "' uses matching options this tokenizer does not implement");
        special[static_cast<std::size_t>(in_vocab->second)] = required_flag(entry, "special");
        added.emplace_back(content, in_vocab->second);
    }
}

// BPE of one word, as the reference merges it: the last character carries the end-of-word
// suffix, an unknown character becomes the unknown token, and the lowest-ranked adjacent
// pair (the leftmost among equal ranks) merges first until no pair has a rule.
void Tokenizer::Impl::encode_word(const std::string& word, const std::vector<std::size_t>& starts,
                                  std::vector<std::int32_t>& out, std::size_t max_tokens) const {
    const auto count = static_cast<std::int32_t>(starts.size());
    std::vector<Symbol> symbols(starts.size());
    std::string piece;
    for (std::int32_t i = 0; i < count; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const auto end = i + 1 < count ? starts[index + 1] : word.size();
        piece.assign(word, starts[index], end - starts[index]);
        if (i + 1 == count)
            piece += suffix;
        const auto found = vocab.find(piece);
        symbols[index] = Symbol{found == vocab.end() ? unk_id : found->second, i - 1,
                                i + 1 < count ? i + 1 : -1, false};
    }
    const auto rule = [&](std::int32_t left, std::int32_t right) -> const MergeRule* {
        const auto found = merges.find(pair_key(left, right));
        return found == merges.end() ? nullptr : &found->second;
    };
    std::priority_queue<Candidate, std::vector<Candidate>, LaterCandidate> queue;
    for (std::int32_t i = 0; i + 1 < count; ++i) {
        const auto index = static_cast<std::size_t>(i);
        if (const auto* merge = rule(symbols[index].id, symbols[index + 1].id))
            queue.push(Candidate{merge->rank, i, merge->merged});
    }
    while (!queue.empty()) {
        const Candidate top = queue.top();
        queue.pop();
        auto& current = symbols[static_cast<std::size_t>(top.pos)];
        if (current.removed || current.next < 0)
            continue;
        auto& right = symbols[static_cast<std::size_t>(current.next)];
        // An entry is stale once either side has merged into something else.
        const auto* merge = rule(current.id, right.id);
        if (merge == nullptr || merge->merged != top.merged)
            continue;
        current.id = top.merged;
        current.next = right.next;
        right.removed = true;
        if (current.next >= 0)
            symbols[static_cast<std::size_t>(current.next)].prev = top.pos;
        if (current.prev >= 0) {
            const auto& previous = symbols[static_cast<std::size_t>(current.prev)];
            if (const auto* left = rule(previous.id, current.id))
                queue.push(Candidate{left->rank, current.prev, left->merged});
        }
        if (current.next >= 0) {
            const auto& following = symbols[static_cast<std::size_t>(current.next)];
            if (const auto* next = rule(current.id, following.id))
                queue.push(Candidate{next->rank, top.pos, next->merged});
        }
    }
    for (const auto& symbol : symbols) {
        if (symbol.removed)
            continue;
        if (out.size() >= max_tokens)
            throw too_many_tokens(max_tokens);
        out.push_back(symbol.id);
    }
}

Tokenizer::Tokenizer(std::string_view tokenizer_json) : impl_(std::make_unique<Impl>()) {
    nlohmann::json json;
    // The parser keeps the last of repeated keys; a repeated key (for example a vocabulary
    // entry) is refused instead.
    std::vector<std::unordered_set<std::string>> keys;
    const auto reject_repeated_keys = [&](int, nlohmann::json::parse_event_t event,
                                          nlohmann::json& parsed) {
        if (event == nlohmann::json::parse_event_t::object_start)
            keys.emplace_back();
        else if (event == nlohmann::json::parse_event_t::object_end)
            keys.pop_back();
        else if (event == nlohmann::json::parse_event_t::key &&
                 !keys.back().insert(parsed.get<std::string>()).second)
            throw contract_error("repeats the key '" + parsed.get<std::string>() + "'");
        return true;
    };
    try {
        json = nlohmann::json::parse(tokenizer_json.begin(), tokenizer_json.end(),
                                     reject_repeated_keys);
    } catch (const nlohmann::json::exception& error) {
        throw contract_error(std::string("is not JSON: ") + error.what());
    }
    try {
        impl_->parse(json);
    } catch (const nlohmann::json::exception& error) {
        // A required field that is missing or has another JSON type.
        throw contract_error(std::string("has an invalid field: ") + error.what());
    }
}

Tokenizer::~Tokenizer() = default;

std::vector<std::int32_t> Tokenizer::encode(std::string_view text, std::size_t max_tokens) const {
    // One pass over the raw text. An added token is matched where it starts, leftmost and
    // then longest; any other code point is normalised and split into words
    // (BertPreTokenizer: whitespace ends a word, punctuation is a word of its own), and
    // each word is encoded as soon as it ends.
    const Impl& impl = *impl_;
    std::vector<std::int32_t> ids;
    std::string word;
    std::vector<std::size_t> starts;
    const auto end_word = [&] {
        if (!starts.empty())
            impl.encode_word(word, starts, ids, max_tokens);
        word.clear();
        starts.clear();
    };
    std::string normalized;
    for (std::size_t pos = 0; pos < text.size();) {
        if (const auto* added = impl.added_at(text, pos)) {
            end_word();
            if (ids.size() >= max_tokens)
                throw too_many_tokens(max_tokens);
            ids.push_back(added->second);
            pos += added->first.size();
            continue;
        }
        normalized.clear();
        normalize_code_point(next_code_point(text, pos), normalized);
        for (std::size_t at = 0; at < normalized.size();) {
            const std::size_t start = at;
            const char32_t cp = next_code_point(normalized, at);
            const bool punctuation = contains(kPunctuation, cp);
            if (punctuation || contains(kWhitespace, cp)) {
                end_word();
                if (!punctuation)
                    continue;
                word.assign(normalized, start, at - start);
                starts.assign(1, 0);
                end_word();
                continue;
            }
            // Stop reading a word that cannot fit in the tokens left.
            if (impl.exceeds(starts.size() + 1, max_tokens - ids.size()))
                throw too_many_tokens(max_tokens);
            starts.push_back(word.size());
            word.append(normalized, start, at - start);
        }
    }
    end_word();
    return ids;
}

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids) const {
    // The reference drops ids without a token and special tokens before decoding, so the
    // last kept token is the one whose suffix becomes nothing instead of a space.
    std::vector<const std::string*> kept;
    for (const auto id : ids) {
        if (id >= 0 && static_cast<std::size_t>(id) < impl_->tokens.size() &&
            !impl_->special[static_cast<std::size_t>(id)])
            kept.push_back(&impl_->tokens[static_cast<std::size_t>(id)]);
    }
    const auto& suffix = impl_->suffix;
    std::string text;
    for (std::size_t index = 0; index < kept.size(); ++index) {
        const std::string& token = *kept[index];
        const char* replacement = index + 1 == kept.size() ? "" : " ";
        std::size_t from = 0;
        for (auto at = token.find(suffix); at != std::string::npos; at = token.find(suffix, from)) {
            text.append(token, from, at - from);
            text.append(replacement);
            from = at + suffix.size();
        }
        text.append(token, from, std::string::npos);
    }
    return text;
}

std::int32_t Tokenizer::vocab_size() const {
    return static_cast<std::int32_t>(impl_->tokens.size());
}

} // namespace trtmc::openai_gpt
