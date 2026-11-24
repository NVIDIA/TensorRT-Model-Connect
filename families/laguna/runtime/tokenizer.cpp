/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/laguna/runtime/tokenizer.h"

#include "families/laguna/runtime/unicode_tables.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <functional>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace trtmc::laguna {

namespace {

constexpr std::string_view kNewlineRunRegex = "(?:\\r?\\n)+(?!\\r?\\n)";
constexpr std::string_view kWordRegex =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// ---- UTF-8 ----

// Decodes one code point; an invalid byte decodes as itself (length 1).
char32_t decode_utf8(std::string_view s, std::size_t i, std::size_t& length) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    const auto continuation = [&](std::size_t k) {
        return i + k < s.size() && (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80;
    };
    if (b0 < 0x80) {
        length = 1;
        return b0;
    }
    if ((b0 & 0xE0) == 0xC0 && continuation(1)) {
        length = 2;
        return ((b0 & 0x1FU) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3FU);
    }
    if ((b0 & 0xF0) == 0xE0 && continuation(1) && continuation(2)) {
        length = 3;
        return ((b0 & 0x0FU) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3FU) << 6) |
               (static_cast<unsigned char>(s[i + 2]) & 0x3FU);
    }
    if ((b0 & 0xF8) == 0xF0 && continuation(1) && continuation(2) && continuation(3)) {
        length = 4;
        return ((b0 & 0x07U) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3FU) << 12) |
               ((static_cast<unsigned char>(s[i + 2]) & 0x3FU) << 6) |
               (static_cast<unsigned char>(s[i + 3]) & 0x3FU);
    }
    length = 1;
    return b0;
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

// Replaces malformed UTF-8 with U+FFFD, as String::from_utf8_lossy.
std::string utf8_lossy(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size());
    for (std::size_t i = 0; i < bytes.size();) {
        const auto lead = static_cast<unsigned char>(bytes[i]);
        const std::size_t length = lead < 0x80                    ? 1
                                   : lead >= 0xC2 && lead <= 0xDF ? 2
                                   : lead >= 0xE0 && lead <= 0xEF ? 3
                                   : lead >= 0xF0 && lead <= 0xF4 ? 4
                                                                  : 0;
        std::size_t valid = length == 0 ? 0 : 1;
        for (std::size_t j = 1; valid == j && j < length && i + j < bytes.size(); ++j) {
            const auto byte = static_cast<unsigned char>(bytes[i + j]);
            bool ok = byte >= 0x80 && byte <= 0xBF;
            if (j == 1)
                ok = ok && !(lead == 0xE0 && byte < 0xA0) && !(lead == 0xED && byte >= 0xA0) &&
                     !(lead == 0xF0 && byte < 0x90) && !(lead == 0xF4 && byte >= 0x90);
            valid += ok ? 1 : 0;
        }
        if (length != 0 && valid == length) {
            out.append(bytes.substr(i, length));
            i += length;
        } else {
            out += "\xEF\xBF\xBD";
            i += std::max<std::size_t>(valid, 1);
        }
    }
    return out;
}

// ---- Character classes of the pre-tokenizer regex ----

template <std::size_t N>
bool in_ranges(char32_t cp, const CodePointRange (&ranges)[N]) {
    const auto* it = std::upper_bound(
        std::begin(ranges), std::end(ranges), cp,
        [](char32_t value, const CodePointRange& range) { return value < range.lo; });
    return it != std::begin(ranges) && cp <= (it - 1)->hi;
}

bool is_letter(char32_t cp) {
    return in_ranges(cp, kLetterRanges);
}
bool is_number(char32_t cp) {
    return in_ranges(cp, kNumberRanges);
}
bool is_crlf(char32_t cp) {
    return cp == '\r' || cp == '\n';
}

// Unicode White_Space, the meaning of \s.
bool is_space(char32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

// [^\s\p{L}\p{N}]
bool is_other(char32_t cp) {
    return !is_space(cp) && !is_letter(cp) && !is_number(cp);
}

char32_t peek(std::string_view s, std::size_t i, std::size_t& length) {
    if (i >= s.size()) {
        length = 0;
        return 0;
    }
    return decode_utf8(s, i, length);
}

template <class Predicate>
std::size_t scan_while(std::string_view s, std::size_t i, Predicate predicate) {
    while (i < s.size()) {
        std::size_t length = 0;
        if (!predicate(decode_utf8(s, i, length)))
            break;
        i += length;
    }
    return i;
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d) after the apostrophe at `i`; 0 when none.
std::size_t contraction_length(std::string_view s, std::size_t i) {
    const auto lower = [&](std::size_t k) {
        return i + k < s.size()
                   ? static_cast<char>(std::tolower(static_cast<unsigned char>(s[i + k])))
                   : '\0';
    };
    const char first = lower(1);
    if (first == 's' || first == 't' || first == 'm' || first == 'd')
        return 2;
    const char second = lower(2);
    if ((first == 'r' && second == 'e') || (first == 'v' && second == 'e') ||
        (first == 'l' && second == 'l'))
        return 3;
    return 0;
}

// End of the leftmost-first match of kWordRegex starting at `i`.
std::size_t word_match_end(std::string_view s, std::size_t i) {
    std::size_t length = 0;
    const char32_t cp = decode_utf8(s, i, length);
    if (cp == '\'') {
        if (const std::size_t contraction = contraction_length(s, i))
            return i + contraction;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    if (is_letter(cp))
        return scan_while(s, i + length, is_letter);
    std::size_t next_length = 0;
    const char32_t next = peek(s, i + length, next_length);
    if (!is_crlf(cp) && !is_number(cp) && next_length != 0 && is_letter(next))
        return scan_while(s, i + length + next_length, is_letter);
    // \p{N}
    if (is_number(cp))
        return i + length;
    // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    const bool spaced_other = cp == ' ' && next_length != 0 && is_other(next);
    if (spaced_other || is_other(cp)) {
        const std::size_t start = spaced_other ? i + length : i;
        return scan_while(s, scan_while(s, start, is_other), is_crlf);
    }
    // Whitespace run [i, run_end).
    const std::size_t run_end = scan_while(s, i, is_space);
    // \s*[\r\n]+ : through the last CR/LF of the run.
    std::size_t last_crlf = std::string_view::npos;
    for (std::size_t k = i; k < run_end; ++k) {
        if (s[k] == '\r' || s[k] == '\n')
            last_crlf = k;
    }
    if (last_crlf != std::string_view::npos)
        return last_crlf + 1;
    // \s+(?!\S) : the whole run at the end, else all but its last character.
    if (run_end == s.size())
        return run_end;
    std::size_t last_start = i;
    for (std::size_t k = i; k < run_end;) {
        last_start = k;
        decode_utf8(s, k, length);
        k += length;
    }
    if (last_start > i)
        return last_start;
    // \s+
    return run_end;
}

// Split points of `(?:\r?\n)+(?!\r?\n)` with MergedWithNext: every newline run
// starts a piece that also holds the text up to the next run.
std::vector<std::string_view> newline_pieces(std::string_view s) {
    const auto unit = [&](std::size_t k) -> std::size_t {
        if (s[k] == '\n')
            return 1;
        return s[k] == '\r' && k + 1 < s.size() && s[k + 1] == '\n' ? 2 : 0;
    };
    std::vector<std::string_view> pieces;
    std::size_t start = 0;
    for (std::size_t k = 0; k < s.size();) {
        std::size_t width = unit(k);
        if (width == 0) {
            ++k;
            continue;
        }
        if (k > start) {
            pieces.push_back(s.substr(start, k - start));
            start = k;
        }
        while (k < s.size() && (width = unit(k)) != 0)
            k += width;
    }
    if (start < s.size())
        pieces.push_back(s.substr(start));
    return pieces;
}

// GPT-2 byte <-> printable code point mapping.
struct ByteTables {
    std::array<std::string, 256> encode;
    std::unordered_map<char32_t, unsigned char> decode;
};

const ByteTables& byte_tables() {
    static const ByteTables tables = [] {
        ByteTables t;
        int extra = 0;
        for (int byte = 0; byte < 256; ++byte) {
            const bool printable = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                                   (byte >= 174 && byte <= 255);
            const char32_t cp =
                printable ? static_cast<char32_t>(byte) : static_cast<char32_t>(256 + extra++);
            append_utf8(t.encode[static_cast<std::size_t>(byte)], cp);
            t.decode[cp] = static_cast<unsigned char>(byte);
        }
        return t;
    }();
    return tables;
}

std::invalid_argument too_many_tokens(std::size_t max_tokens) {
    return std::invalid_argument("Laguna prompt has more than " + std::to_string(max_tokens) +
                                 " tokens");
}

// Calls visit(word) for each pre-tokenizer piece of one segment without added tokens, in order.
template <class Visit>
void for_each_word(std::string_view text, Visit&& visit) {
    for (const auto piece : newline_pieces(text)) {
        for (std::size_t k = 0; k < piece.size();) {
            const std::size_t end = std::max(word_match_end(piece, k), k + 1);
            visit(piece.substr(k, end - k));
            k = end;
        }
    }
}

struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& pair) const {
        const std::size_t a = std::hash<std::string>{}(pair.first);
        const std::size_t b = std::hash<std::string>{}(pair.second);
        return a ^ (b + 0x9E3779B97F4A7C15ULL + (a << 6) + (a >> 2));
    }
};

const nlohmann::json& require(const nlohmann::json& object, const char* key) {
    if (!object.is_object() || !object.contains(key))
        throw std::runtime_error(std::string("laguna tokenizer.json is missing '") + key + "'");
    return object.at(key);
}

void require_equal(const nlohmann::json& object, const char* key, const nlohmann::json& value) {
    if (require(object, key) != value)
        throw std::runtime_error(std::string("laguna tokenizer.json has unsupported '") + key +
                                 "': " + object.at(key).dump());
}

// A token id is an integer in [0, INT32_MAX]: get<std::int32_t>() would truncate 1.5 and wrap 2^32.
std::int32_t token_id(const nlohmann::json& value) {
    if (!value.is_number_integer() || value.get<std::int64_t>() < 0 ||
        value.get<std::int64_t>() > INT32_MAX)
        throw std::runtime_error("laguna tokenizer.json has an invalid token id: " + value.dump());
    return static_cast<std::int32_t>(value.get<std::int64_t>());
}

void validate_split(const nlohmann::json& split, std::string_view regex, const char* behavior) {
    require_equal(split, "type", "Split");
    require_equal(split, "behavior", behavior);
    require_equal(split, "invert", false);
    if (require(require(split, "pattern"), "Regex").get<std::string>() != regex)
        throw std::runtime_error("laguna tokenizer.json has an unsupported Split pattern");
}

} // namespace

struct Tokenizer::State {
    std::unordered_map<std::string, std::int32_t> vocab;
    std::unordered_map<std::int32_t, std::string> id_to_token;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_rank;
    std::vector<std::pair<std::string, std::int32_t>> added;
    std::unordered_set<std::int32_t> special;
    std::vector<std::int32_t> prefix_ids;
    std::vector<std::int32_t> suffix_ids;
    // Bytes of the longest vocabulary or added token: no token covers more input bytes.
    std::size_t longest_token = 1;
    std::int32_t largest_id = -1;

    void parse(const nlohmann::json& json);
    void encode_word(std::string_view word, std::vector<std::int32_t>& out,
                     std::size_t max_tokens) const;
    void encode_text(std::string_view text, std::vector<std::int32_t>& out,
                     std::size_t max_tokens) const;
};

void Tokenizer::State::parse(const nlohmann::json& json) {
    const auto& model = require(json, "model");
    require_equal(model, "type", "BPE");
    require_equal(model, "byte_fallback", false);
    require_equal(model, "ignore_merges", false);
    require_equal(model, "continuing_subword_prefix", "");
    require_equal(model, "end_of_word_suffix", "");
    // BPE dropout skips merges at random; only deterministic merging (no or zero dropout) is
    // supported.
    if (model.contains("dropout")) {
        const auto& dropout = model.at("dropout");
        if (!dropout.is_null() && !(dropout.is_number() && dropout.get<double>() == 0.0))
            throw std::runtime_error("laguna tokenizer.json BPE dropout is not supported: " +
                                     dropout.dump());
    }
    if (!require(json, "normalizer").is_null())
        throw std::runtime_error("laguna tokenizer.json must not declare a normalizer");

    const auto& pre = require(json, "pre_tokenizer");
    require_equal(pre, "type", "Sequence");
    const auto& steps = require(pre, "pretokenizers");
    if (!steps.is_array() || steps.size() != 3)
        throw std::runtime_error("laguna tokenizer.json must have three pre-tokenizers");
    validate_split(steps[0], kNewlineRunRegex, "MergedWithNext");
    validate_split(steps[1], kWordRegex, "Isolated");
    require_equal(steps[2], "type", "ByteLevel");
    require_equal(steps[2], "add_prefix_space", false);
    require_equal(steps[2], "use_regex", false);
    require_equal(require(json, "decoder"), "type", "ByteLevel");

    for (const auto& [token, id] : require(model, "vocab").items()) {
        const auto value = token_id(id);
        if (!id_to_token.emplace(value, token).second)
            throw std::runtime_error("laguna tokenizer.json repeats vocab id " +
                                     std::to_string(value));
        vocab[token] = value;
        longest_token = std::max(longest_token, token.size());
        largest_id = std::max(largest_id, value);
    }
    const auto& merges = require(model, "merges");
    if (!merges.is_array() || merges.size() > static_cast<std::size_t>(INT_MAX))
        throw std::runtime_error(
            "laguna tokenizer.json merges must be a list of at most INT_MAX rules");
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        const auto& merge = merges[rank];
        if (!merge.is_array() || merge.size() != 2 || !merge[0].is_string() ||
            !merge[1].is_string())
            throw std::runtime_error("laguna tokenizer.json merges must be string pairs");
        auto first = merge[0].get<std::string>();
        auto second = merge[1].get<std::string>();
        // Both operands and their concatenation are vocabulary tokens, as tokenizers requires.
        if (vocab.count(first) == 0 || vocab.count(second) == 0 || vocab.count(first + second) == 0)
            throw std::runtime_error("laguna tokenizer.json merge '" + first + " " + second +
                                     "' is not within the vocabulary");
        if (!merge_rank
                 .emplace(std::make_pair(std::move(first), std::move(second)),
                          static_cast<int>(rank))
                 .second)
            throw std::runtime_error("laguna tokenizer.json repeats a merge rule");
    }
    // tokenizers keeps a vocabulary token's id for an added token and numbers the others from the
    // vocabulary size on (after the largest added id once that reaches it); a declared id must
    // equal that assignment.
    const auto vocab_size = static_cast<std::int64_t>(vocab.size());
    std::int64_t largest_added = -1;
    for (const auto& token : require(json, "added_tokens")) {
        for (const char* flag : {"lstrip", "rstrip", "single_word", "normalized"})
            require_equal(token, flag, false);
        const auto content = require(token, "content").get<std::string>();
        // An empty added token would match at every position without advancing encode().
        if (content.empty())
            throw std::runtime_error("laguna tokenizer.json has an added token with empty content");
        const auto id = token_id(require(token, "id"));
        const auto known_id = id_to_token.find(id);
        const auto known_content = vocab.find(content);
        const std::int64_t assigned = known_content != vocab.end()  ? known_content->second
                                      : largest_added >= vocab_size ? largest_added + 1
                                                                    : vocab_size;
        if (id != assigned || (known_id != id_to_token.end() && known_id->second != content))
            throw std::runtime_error("laguna tokenizer.json added token '" + content + "' has id " +
                                     std::to_string(id) + "; tokenizers assigns " +
                                     std::to_string(assigned));
        largest_added = std::max<std::int64_t>(largest_added, id);
        added.emplace_back(content, id);
        id_to_token[id] = content;
        vocab[content] = id;
        longest_token = std::max(longest_token, content.size());
        largest_id = std::max(largest_id, id);
        if (require(token, "special").get<bool>())
            special.insert(id);
    }
    // Longest content first: the first hit at a position is the longest match.
    std::stable_sort(added.begin(), added.end(),
                     [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    const auto& post = require(json, "post_processor");
    require_equal(post, "type", "TemplateProcessing");
    // Each item is one operation; the input is one Sequence A, with special tokens before or after
    // it.
    bool after_sequence = false;
    for (const auto& item : require(post, "single")) {
        if (!item.is_object() || item.size() != 1)
            throw std::runtime_error(
                "laguna tokenizer.json post-processor items hold exactly one operation");
        if (item.contains("Sequence")) {
            if (after_sequence || require(item.at("Sequence"), "id") != "A")
                throw std::runtime_error(
                    "laguna tokenizer.json post-processor needs exactly one Sequence A");
            after_sequence = true;
            continue;
        }
        const auto name = require(require(item, "SpecialToken"), "id").get<std::string>();
        const auto& ids = require(require(require(post, "special_tokens"), name.c_str()), "ids");
        for (const auto& id : ids) {
            (after_sequence ? suffix_ids : prefix_ids).push_back(token_id(id));
            largest_id = std::max(largest_id, token_id(id));
        }
    }
    if (!after_sequence)
        throw std::runtime_error(
            "laguna tokenizer.json post-processor needs exactly one Sequence A");
}

void Tokenizer::State::encode_word(std::string_view word, std::vector<std::int32_t>& out,
                                   std::size_t max_tokens) const {
    // tokenizers' BPE: merge one pair occurrence at a time, the lowest rank first and the leftmost
    // among equal ranks, then reconsider the merged symbol's neighbours. Merging every occurrence
    // of a pair at once gives other tokens when a merge creates a pair of lower rank.
    struct Symbol {
        std::string text; // empty once merged into its left neighbour
        std::ptrdiff_t prev;
        std::ptrdiff_t next;
    };
    const auto& tables = byte_tables();
    const auto length = static_cast<std::ptrdiff_t>(word.size());
    std::vector<Symbol> symbols;
    symbols.reserve(word.size());
    for (std::ptrdiff_t k = 0; k < length; ++k)
        symbols.push_back(
            {tables.encode[static_cast<unsigned char>(word[static_cast<std::size_t>(k)])], k - 1,
             k + 1 < length ? k + 1 : -1});
    using Candidate = std::pair<int, std::ptrdiff_t>; // (rank, left position)
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;
    const auto consider = [&](std::ptrdiff_t left) {
        const auto right = symbols[static_cast<std::size_t>(left)].next;
        if (right < 0)
            return;
        const auto it = merge_rank.find({symbols[static_cast<std::size_t>(left)].text,
                                         symbols[static_cast<std::size_t>(right)].text});
        if (it != merge_rank.end())
            queue.push({it->second, left});
    };
    for (std::ptrdiff_t k = 0; k + 1 < length; ++k)
        consider(k);
    while (!queue.empty()) {
        const auto [rank, left] = queue.top();
        queue.pop();
        auto& symbol = symbols[static_cast<std::size_t>(left)];
        if (symbol.text.empty() || symbol.next < 0)
            continue;
        auto& right = symbols[static_cast<std::size_t>(symbol.next)];
        // A candidate is stale once either symbol changed: ranks identify rules, and rules are
        // unique.
        const auto it = merge_rank.find({symbol.text, right.text});
        if (it == merge_rank.end() || it->second != rank)
            continue;
        symbol.text += right.text;
        right.text.clear();
        symbol.next = right.next;
        if (symbol.next >= 0)
            symbols[static_cast<std::size_t>(symbol.next)].prev = left;
        if (symbol.prev >= 0)
            consider(symbol.prev);
        consider(left);
    }
    for (const auto& symbol : symbols) {
        if (symbol.text.empty())
            continue;
        const auto it = vocab.find(symbol.text);
        if (it == vocab.end())
            throw std::runtime_error("laguna tokenizer produced a symbol outside its vocabulary");
        if (out.size() >= max_tokens)
            throw too_many_tokens(max_tokens);
        out.push_back(it->second);
    }
}

void Tokenizer::State::encode_text(std::string_view text, std::vector<std::int32_t>& out,
                                   std::size_t max_tokens) const {
    for_each_word(text, [&](std::string_view word) {
        // A word of n bytes has at least ceil(n / longest_token) tokens: reject it before merging.
        const std::size_t least = (word.size() + longest_token - 1) / longest_token;
        if (out.size() > max_tokens || least > max_tokens - out.size())
            throw too_many_tokens(max_tokens);
        encode_word(word, out, max_tokens);
    });
}

Tokenizer::Tokenizer(std::unique_ptr<State> state) : state_(std::move(state)) {}
Tokenizer::~Tokenizer() = default;

std::unique_ptr<Tokenizer> Tokenizer::create(std::string_view tokenizer_json) {
    nlohmann::json json;
    // The parser keeps the last of repeated keys; a repeated key (for example a vocab entry) is
    // rejected.
    std::vector<std::unordered_set<std::string>> keys;
    const auto reject_repeated_keys = [&](int, nlohmann::json::parse_event_t event,
                                          nlohmann::json& parsed) {
        if (event == nlohmann::json::parse_event_t::object_start)
            keys.emplace_back();
        else if (event == nlohmann::json::parse_event_t::object_end)
            keys.pop_back();
        else if (event == nlohmann::json::parse_event_t::key &&
                 !keys.back().insert(parsed.get<std::string>()).second)
            throw std::runtime_error("laguna tokenizer.json repeats the key '" +
                                     parsed.get<std::string>() + "'");
        return true;
    };
    try {
        json = nlohmann::json::parse(tokenizer_json.begin(), tokenizer_json.end(),
                                     reject_repeated_keys);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("laguna cannot parse tokenizer.json: ") +
                                 error.what());
    }
    auto state = std::make_unique<State>();
    state->parse(json);
    return std::unique_ptr<Tokenizer>(new Tokenizer(std::move(state)));
}

std::int32_t Tokenizer::max_id() const {
    return state_->largest_id;
}

std::size_t Tokenizer::max_text_bytes(std::size_t max_tokens) const {
    return max_tokens < SIZE_MAX / state_->longest_token ? max_tokens * state_->longest_token
                                                         : SIZE_MAX;
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text, bool add_special_tokens,
                                            std::size_t max_tokens) const {
    // A text longer than max_text_bytes(max_tokens) has more tokens than the budget.
    if (text.size() > max_text_bytes(max_tokens))
        throw too_many_tokens(max_tokens);
    std::vector<std::int32_t> ids;
    if (add_special_tokens) {
        if (state_->prefix_ids.size() > max_tokens)
            throw too_many_tokens(max_tokens);
        ids = state_->prefix_ids;
    }
    std::size_t plain = 0;
    for (std::size_t k = 0; k < text.size();) {
        const auto hit =
            std::find_if(state_->added.begin(), state_->added.end(), [&](const auto& a) {
                return text.compare(k, a.first.size(), a.first) == 0;
            });
        if (hit == state_->added.end()) {
            ++k;
            continue;
        }
        state_->encode_text(text.substr(plain, k - plain), ids, max_tokens);
        if (ids.size() >= max_tokens)
            throw too_many_tokens(max_tokens);
        ids.push_back(hit->second);
        k += hit->first.size();
        plain = k;
    }
    state_->encode_text(text.substr(plain), ids, max_tokens);
    if (add_special_tokens) {
        if (state_->suffix_ids.size() > max_tokens - ids.size())
            throw too_many_tokens(max_tokens);
        ids.insert(ids.end(), state_->suffix_ids.begin(), state_->suffix_ids.end());
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids) const {
    const auto& tables = byte_tables();
    std::string bytes;
    for (const std::int32_t id : ids) {
        if (state_->special.count(id) != 0)
            continue;
        const auto it = state_->id_to_token.find(id);
        if (it == state_->id_to_token.end())
            continue;
        // ByteLevel decoding maps a token's characters back to bytes; a token
        // with any character outside the byte alphabet is emitted verbatim.
        std::string mapped;
        bool all_mapped = true;
        for (std::size_t k = 0; k < it->second.size() && all_mapped;) {
            std::size_t length = 0;
            const char32_t cp = decode_utf8(it->second, k, length);
            const auto byte = tables.decode.find(cp);
            all_mapped = byte != tables.decode.end();
            if (all_mapped)
                mapped.push_back(static_cast<char>(byte->second));
            k += length;
        }
        bytes += all_mapped ? mapped : it->second;
    }
    return utf8_lossy(bytes);
}

std::int32_t Tokenizer::id_for_token(std::string_view token) const {
    const auto it = state_->vocab.find(std::string(token));
    return it == state_->vocab.end() ? -1 : it->second;
}

std::vector<std::string> pre_tokenize(std::string_view text) {
    std::vector<std::string> words;
    for_each_word(text, [&](std::string_view word) { words.emplace_back(word); });
    return words;
}

} // namespace trtmc::laguna
