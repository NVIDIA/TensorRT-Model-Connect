/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deberta_v2/runtime/tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

// ─── UTF-8 helpers ───

inline char32_t utf8_to_char32(const std::string& s, size_t& pos) {
    unsigned char c = static_cast<unsigned char>(s[pos]);
    if (c < 0x80) {
        ++pos;
        return static_cast<char32_t>(c);
    }
    if ((c & 0xE0) == 0xC0 && pos + 1 < s.size()) {
        char32_t cp = (static_cast<char32_t>(c & 0x1F) << 6) |
                      static_cast<char32_t>(static_cast<unsigned char>(s[pos + 1]) & 0x3F);
        pos += 2;
        return cp;
    }
    if ((c & 0xF0) == 0xE0 && pos + 2 < s.size()) {
        char32_t cp = (static_cast<char32_t>(c & 0x0F) << 12) |
                      (static_cast<char32_t>(static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 6) |
                      static_cast<char32_t>(static_cast<unsigned char>(s[pos + 2]) & 0x3F);
        pos += 3;
        return cp;
    }
    if ((c & 0xF8) == 0xF0 && pos + 3 < s.size()) {
        char32_t cp = (static_cast<char32_t>(c & 0x07) << 18) |
                      (static_cast<char32_t>(static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 12) |
                      (static_cast<char32_t>(static_cast<unsigned char>(s[pos + 2]) & 0x3F) << 6) |
                      static_cast<char32_t>(static_cast<unsigned char>(s[pos + 3]) & 0x3F);
        pos += 4;
        return cp;
    }
    ++pos;
    return 0xFFFD;
}

inline std::string char32_to_utf8(char32_t cp) {
    std::string r;
    if (cp <= 0x7F) {
        r.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        r.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        r.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        r.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        r.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        r.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0x10FFFF) {
        r.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        r.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        r.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        r.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return r;
}

// Return the byte length of the UTF-8 codepoint starting at s[pos].
inline size_t utf8_char_len(const std::string& s, size_t pos) {
    unsigned char c = static_cast<unsigned char>(s[pos]);
    if (c < 0x80)
        return 1;
    if ((c & 0xE0) == 0xC0)
        return 2;
    if ((c & 0xF0) == 0xE0)
        return 3;
    if ((c & 0xF8) == 0xF0)
        return 4;
    return 1;
}

// ─── Precompiled Normalizer ───
//
// The Precompiled charsmap is a binary blob from HuggingFace tokenizers:
//   [4 bytes LE] trie_size
//   [trie_size bytes] double-array trie for NFKC normalization
//   [remaining bytes] normalized string pool
//
// For simplicity, we skip the full NFKC trie and only handle:
// 1. Control char removal (U+0000-U+001F except tab/newline/CR)
// 2. Whitespace normalization (various Unicode spaces → regular space)
// This is sufficient for most practical text inputs.

inline bool is_control(char32_t cp) {
    if (cp == '\t' || cp == '\n' || cp == '\r')
        return false;
    return (cp < 0x20) || cp == 0x7F || (cp >= 0x80 && cp <= 0x9F);
}

struct UnicodeRange {
    char32_t lo, hi;
};
constexpr UnicodeRange kUnicodeSpaces[] = {
    {0x00A0, 0x00A0}, {0x1680, 0x1680}, {0x2000, 0x200A}, {0x2028, 0x2029},
    {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000},
};

inline bool is_unicode_space(char32_t cp) {
    for (const auto& r : kUnicodeSpaces)
        if (cp >= r.lo && cp <= r.hi)
            return true;
    return false;
}

std::string precompiled_normalize(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    size_t pos = 0;
    while (pos < text.size()) {
        char32_t cp = utf8_to_char32(text, pos);
        if (is_control(cp))
            continue;
        if (is_unicode_space(cp)) {
            result += ' ';
            continue;
        }
        result += char32_to_utf8(cp);
    }
    return result;
}

struct NormalizerStep {
    enum class Kind {
        Precompiled,
        Lowercase,
        DropChars,
        CollapseWhitespace,
        CollapseSpaces,
        CollapseWhitespaceRuns,
        Nfc,
        Strip,
        StripRight
    };
    Kind kind{Kind::Lowercase};
    // Only DropChars uses this: the literal characters its pattern names.
    std::string chars;
};

// Expand a regex character class such as `[!"\#\$...]` into the literal
// characters it matches. Only the one shape SigLIP uses is accepted - a flat
// class of single characters, where a backslash quotes the next one - and
// anything else is refused rather than approximated. Deriving the set from
// the pattern keeps one source of truth: a hand-copied second list of
// punctuation would be a silent place for the two to drift apart.
std::string expand_character_class(const std::string& regex) {
    if (regex.size() < 3 || regex.front() != '[' || regex.back() != ']')
        throw std::runtime_error("deberta_v2 tokenizer: not a character class: " + regex);
    std::string chars;
    for (std::size_t i = 1; i + 1 < regex.size(); ++i) {
        char c = regex[i];
        if (c == '\\') {
            if (i + 2 >= regex.size())
                throw std::runtime_error("deberta_v2 tokenizer: dangling escape in " + regex);
            c = regex[++i];
        } else if (c == '-' || c == '^') {
            // A range or a negation is a different language from a flat list.
            throw std::runtime_error("deberta_v2 tokenizer: unsupported class operator in " +
                                     regex);
        }
        if (static_cast<unsigned char>(c) >= 0x80)
            throw std::runtime_error("deberta_v2 tokenizer: non-ASCII class member in " + regex);
        chars.push_back(c);
    }
    return chars;
}

std::string drop_chars(std::string text, const std::string& chars) {
    std::string out;
    out.reserve(text.size());
    for (char c : text) {
        if (chars.find(c) == std::string::npos)
            out.push_back(c);
    }
    return out;
}

std::string collapse_whitespace(std::string text) {
    std::string out;
    out.reserve(text.size());
    bool in_space = false;
    for (char c : text) {
        const bool space = std::isspace(static_cast<unsigned char>(c)) != 0;
        if (space) {
            if (!in_space)
                out.push_back(' ');
            in_space = true;
            continue;
        }
        in_space = false;
        out.push_back(c);
    }
    return out;
}

// ` {2,}` is not the same rule as `\s+`: it collapses runs of the literal
// space only, and it is applied after the Precompiled map rather than before.
std::string collapse_spaces(std::string text) {
    std::string out;
    out.reserve(text.size());
    std::size_t run = 0;
    for (char c : text) {
        if (c == ' ') {
            ++run;
            continue;
        }
        if (run > 0)
            out.push_back(' ');
        run = 0;
        out.push_back(c);
    }
    if (run > 0)
        out.push_back(' ');
    return out;
}

std::string strip_outer_whitespace(std::string text) {
    std::size_t first = 0;
    while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first])) != 0)
        ++first;
    std::size_t last = text.size();
    while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1])) != 0)
        --last;
    return text.substr(first, last - first);
}

// `\s{2,}|[\n\r\t]` -> " ". Not the same rule as `\s+` -> " ": a lone space
// is left alone and so is a lone form feed or vertical tab, because neither
// alternative matches one of those on its own. Only runs of two or more, and
// single tabs, newlines and carriage returns, become one space.
std::string collapse_whitespace_runs(std::string text) {
    const auto is_space = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
    std::string out;
    out.reserve(text.size());
    std::size_t index = 0;
    while (index < text.size()) {
        if (!is_space(text[index])) {
            out.push_back(text[index]);
            ++index;
            continue;
        }
        std::size_t end = index;
        while (end < text.size() && is_space(text[end]))
            ++end;
        const std::size_t run = end - index;
        const char single = text[index];
        if (run >= 2 || single == '\n' || single == '\r' || single == '\t')
            out.push_back(' ');
        else
            out.push_back(single);
        index = end;
    }
    return out;
}

// NFC, as far as this tokenizer can honestly go.
//
// Implementing Unicode canonical composition in full is a large table, and
// getting it subtly wrong would change tokenization silently. What is done
// instead is to refuse the input that NFC would actually have changed in
// practice: text carrying combining marks, which is what decomposed accents
// look like. Text without them is already composed, and NFC is the identity
// on it, so passing it through unchanged is exact rather than approximate.
//
// The residual is narrow and real: a few singletons that NFC rewrites on
// their own, such as U+212B ANGSTROM SIGN, carry no combining mark and are
// passed through here where the reference would have replaced them.
bool is_combining_mark(char32_t code) {
    return (code >= 0x0300 && code <= 0x036F) || (code >= 0x1AB0 && code <= 0x1AFF) ||
           (code >= 0x1DC0 && code <= 0x1DFF) || (code >= 0x20D0 && code <= 0x20FF) ||
           (code >= 0xFE20 && code <= 0xFE2F);
}

std::string require_composed(std::string text) {
    std::size_t pos = 0;
    while (pos < text.size()) {
        const char32_t code = utf8_to_char32(text, pos);
        if (is_combining_mark(code))
            throw std::runtime_error(
                "deberta_v2 tokenizer: input carries combining marks, which this tokenizer "
                "cannot compose; supply NFC-normalised text");
    }
    return text;
}

std::string strip_trailing_whitespace(std::string text) {
    std::size_t last = text.size();
    while (last > 0 && std::isspace(static_cast<unsigned char>(text[last - 1])) != 0)
        --last;
    text.resize(last);
    return text;
}

std::string lowercase_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// ─── Metaspace Pre-tokenizer ───
// Replaces spaces with ▁ (U+2581) and optionally adds prefix space.

static const std::string kMetaspaceChar = "\xe2\x96\x81"; // ▁ U+2581

inline bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::vector<std::string> whitespace_split(const std::string& text) {
    std::vector<std::string> words;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && is_ws(text[i]))
            ++i;
        if (i >= text.size())
            break;
        size_t start = i;
        while (i < text.size() && !is_ws(text[i]))
            ++i;
        words.push_back(text.substr(start, i - start));
    }
    return words;
}

[[maybe_unused]] std::string metaspace_pre_tokenize(const std::string& text,
                                                    bool add_prefix_space) {
    std::string result;
    result.reserve(text.size() + 8);
    if (add_prefix_space && !text.empty() && text[0] != ' ') {
        result += kMetaspaceChar;
    }
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == ' ') {
            result += kMetaspaceChar;
        } else {
            result += text[i];
        }
    }
    return result;
}

// ─── Trie for efficient vocab prefix lookup ───

struct TrieNode {
    std::unordered_map<char, int> children;
    int token_id = -1; // -1 = not a token end
    float score = 0.0f;
};

class Trie {
  public:
    Trie() { mNodes.emplace_back(); } // root node

    void insert(const std::string& token, int id, float score) {
        int node = 0;
        for (char c : token) {
            auto it = mNodes[node].children.find(c);
            if (it == mNodes[node].children.end()) {
                int next = static_cast<int>(mNodes.size());
                mNodes.emplace_back();
                mNodes[node].children[c] = next;
                node = next;
            } else {
                node = it->second;
            }
        }
        mNodes[node].token_id = id;
        mNodes[node].score = score;
    }

    // Find all tokens that match a prefix of text starting at offset.
    // Returns vector of (token_id, byte_length, score).
    struct Match {
        int token_id;
        size_t length;
        float score;
    };

    void find_prefixes(const std::string& text, size_t offset, std::vector<Match>& out) const {
        out.clear();
        int node = 0;
        for (size_t i = offset; i < text.size(); ++i) {
            char c = text[i];
            auto it = mNodes[node].children.find(c);
            if (it == mNodes[node].children.end())
                break;
            node = it->second;
            if (mNodes[node].token_id >= 0) {
                out.push_back({mNodes[node].token_id, i - offset + 1, mNodes[node].score});
            }
        }
    }

  private:
    std::vector<TrieNode> mNodes;
};

// ─── Viterbi algorithm for Unigram tokenization ───

struct ViterbiNode {
    float score;
    int token_id;
    size_t prev_pos; // byte position of previous node
};

std::vector<int32_t> viterbi_encode(const std::string& text, const Trie& trie, int32_t unk_id,
                                    float unk_score) {
    if (text.empty())
        return {};

    size_t n = text.size();
    // best[i] = best path score to reach byte position i
    std::vector<ViterbiNode> best(n + 1, {-std::numeric_limits<float>::infinity(), -1, 0});
    best[0].score = 0.0f;

    std::vector<Trie::Match> matches;

    for (size_t i = 0; i < n; ++i) {
        if (best[i].score == -std::numeric_limits<float>::infinity())
            continue;

        trie.find_prefixes(text, i, matches);

        for (const auto& m : matches) {
            size_t end = i + m.length;
            float new_score = best[i].score + m.score;
            if (new_score > best[end].score) {
                best[end].score = new_score;
                best[end].token_id = m.token_id;
                best[end].prev_pos = i;
            }
        }

        // Advance by one UTF-8 codepoint after emitting UNK.
        size_t char_len = utf8_char_len(text, i);
        size_t end = i + char_len;
        if (end <= n) {
            float new_score = best[i].score + unk_score;
            if (new_score > best[end].score) {
                best[end].score = new_score;
                best[end].token_id = unk_id;
                best[end].prev_pos = i;
            }
        }
    }

    // Backtrack to find the best path
    std::vector<int32_t> ids;
    size_t pos = n;
    while (pos > 0) {
        if (best[pos].token_id < 0) {
            // Unreachable after emitting UNK.
            ids.push_back(unk_id);
            break;
        }
        ids.push_back(best[pos].token_id);
        pos = best[pos].prev_pos;
    }

    std::reverse(ids.begin(), ids.end());
    return ids;
}

// ─── UnigramTokenizer ───

class UnigramTokenizer final : public ITokenizer {
  public:
    static std::unique_ptr<UnigramTokenizer> Create(const char* json_data, std::size_t json_size,
                                                    bool add_special_tokens) {
        auto tok = std::unique_ptr<UnigramTokenizer>(new UnigramTokenizer());
        tok->mAddSpecialTokens = add_special_tokens;
        tok->parse_tokenizer_json(json_data, json_size);
        return tok;
    }

    std::vector<int32_t> encode(const std::string& text) const override {
        if (text.empty()) {
            return mAddSpecialTokens ? make_special_frame({}) : std::vector<int32_t>{};
        }

        // Normalize, in the order the checkpoint declares.
        std::string normalized = normalize(text);

        // Pre-tokenize: WhitespaceSplit → Metaspace
        auto words = whitespace_split(normalized);
        std::vector<int32_t> ids;
        for (size_t i = 0; i < words.size(); ++i) {
            // Metaspace: prepend ▁ to each word
            std::string processed = kMetaspaceChar + words[i];
            auto word_ids = viterbi_encode(processed, mTrie, mUnkId, mUnkScore);
            ids.insert(ids.end(), word_ids.begin(), word_ids.end());
        }

        if (mAddSpecialTokens)
            ids = make_special_frame(ids);
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string result;
        for (int32_t id : ids) {
            if (mDecodeSkipIds.count(id))
                continue;
            std::string token = token_for_id(id);
            result += token;
        }
        // Remove metaspace characters and clean up
        return decode_metaspace(result);
    }

    int32_t id_for_token(std::string_view token) const override {
        auto it = mTokenToId.find(std::string(token));
        return it != mTokenToId.end() ? it->second : -1;
    }

    std::string token_for_id(int32_t id) const override {
        if (id >= 0 && static_cast<size_t>(id) < mIdToToken.size()) {
            return mIdToToken[id];
        }
        return "";
    }

  private:
    UnigramTokenizer() = default;

    static std::string decode_metaspace(const std::string& text) {
        std::string result;
        size_t pos = 0;
        while (pos < text.size()) {
            if (pos + kMetaspaceChar.size() <= text.size() &&
                text.compare(pos, kMetaspaceChar.size(), kMetaspaceChar) == 0) {
                if (!result.empty())
                    result += ' ';
                pos += kMetaspaceChar.size();
            } else {
                result += text[pos];
                ++pos;
            }
        }
        return result;
    }

    std::vector<int32_t> make_special_frame(std::vector<int32_t> ids) const {
        std::vector<int32_t> result;
        if (mBosId >= 0)
            result.push_back(mBosId);
        result.insert(result.end(), ids.begin(), ids.end());
        if (mEosId >= 0)
            result.push_back(mEosId);
        return result;
    }

    // ─── JSON parsing ───

    void parse_tokenizer_json(const char* json_data, std::size_t json_size) {
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(json_data, json_data + json_size);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse tokenizer.json: ") + e.what());
        }

        validate_model(j);
        parse_vocab(j);
        build_trie();
        parse_normalizer(j);
        parse_pre_tokenizer(j);
        parse_added_tokens(j);
        parse_post_processor(j);
        resolve_special_ids();
    }

    static void validate_model(const nlohmann::json& j) {
        if (!j.contains("model"))
            throw std::runtime_error("Invalid tokenizer.json: missing model");

        auto& model = j["model"];

        if (!model.contains("vocab") || !model["vocab"].is_array())
            throw std::runtime_error("Invalid tokenizer.json: model.vocab must be an array");
    }

    void parse_vocab(const nlohmann::json& j) {
        auto& vocab = j["model"]["vocab"];
        mIdToToken.resize(vocab.size());
        mUnkId = j["model"].value("unk_id", 0);

        for (size_t i = 0; i < vocab.size(); ++i) {
            auto& entry = vocab[i];
            std::string token = entry[0].get<std::string>();
            float score = entry[1].get<float>();

            mIdToToken[i] = token;
            mTokenToId[token] = static_cast<int32_t>(i);
            mScores.push_back(score);
        }

        // UNK score: must be worse than ANY real vocab token for Viterbi
        float min_score = 0.0f;
        for (float s : mScores) {
            if (s < min_score)
                min_score = s;
        }
        mUnkScore = min_score - 10.0f;
    }

    void build_trie() {
        for (size_t i = 0; i < mIdToToken.size(); ++i) {
            const auto& token = mIdToToken[i];
            if (!token.empty()) {
                mTrie.insert(token, static_cast<int>(i), mScores[i]);
            }
        }
    }

    // SigLIP's normalizer is an ordered Sequence of six steps, and the order
    // is load-bearing: the punctuation strip turns characters into nothing
    // before whitespace is collapsed, and a second space-collapse runs after
    // the Precompiled map because that map can introduce spaces of its own.
    // The steps are therefore replayed in the declared order rather than
    // reduced to booleans this file applies in an order of its own choosing.
    //
    // An unrecognised step is refused. A tokenizer that quietly skips a
    // normalisation it does not implement still returns ids, and they are
    // wrong only for the inputs that step existed to handle.
    void apply_normalizer_config(const nlohmann::json& norm) {
        const std::string ntype = norm.value("type", "");
        if (ntype == "Precompiled") {
            mSteps.push_back({NormalizerStep::Kind::Precompiled, ""});
            return;
        }
        if (ntype == "Lowercase") {
            mSteps.push_back({NormalizerStep::Kind::Lowercase, ""});
            return;
        }
        if (ntype == "NFC") {
            mSteps.push_back({NormalizerStep::Kind::Nfc, ""});
            return;
        }
        if (ntype == "Strip") {
            const bool left = norm.value("strip_left", true);
            const bool right = norm.value("strip_right", true);
            if (left && right) {
                mSteps.push_back({NormalizerStep::Kind::Strip, ""});
                return;
            }
            if (right) {
                mSteps.push_back({NormalizerStep::Kind::StripRight, ""});
                return;
            }
            throw std::runtime_error("deberta_v2 tokenizer: left-only Strip is not supported");
        }
        if (ntype == "Replace") {
            const auto& pattern = norm.at("pattern");
            if (!pattern.contains("Regex"))
                throw std::runtime_error("deberta_v2 tokenizer: Replace needs a Regex pattern");
            const auto regex = pattern["Regex"].get<std::string>();
            const auto content = norm.value("content", std::string());
            if (content.empty()) {
                mSteps.push_back({NormalizerStep::Kind::DropChars, expand_character_class(regex)});
                return;
            }
            if (content == " " && regex == "\\s+") {
                mSteps.push_back({NormalizerStep::Kind::CollapseWhitespace, ""});
                return;
            }
            if (content == " " && regex == " {2,}") {
                mSteps.push_back({NormalizerStep::Kind::CollapseSpaces, ""});
                return;
            }
            if (content == " " && regex == "\\s{2,}|[\\n\\r\\t]") {
                mSteps.push_back({NormalizerStep::Kind::CollapseWhitespaceRuns, ""});
                return;
            }
            throw std::runtime_error("deberta_v2 tokenizer: unsupported Replace " + regex +
                                     " -> '" + content + "'");
        }
        if (ntype == "Prepend") {
            mAddPrefixSpace = true;
            return;
        }
        throw std::runtime_error("deberta_v2 tokenizer: unsupported normalizer: " + ntype);
    }

    void parse_normalizer(const nlohmann::json& j) {
        if (!j.contains("normalizer") || j["normalizer"].is_null())
            return;
        const auto& norm = j["normalizer"];
        if (norm.value("type", "") == "Sequence" && norm.contains("normalizers")) {
            for (const auto& sub : norm["normalizers"])
                apply_normalizer_config(sub);
            return;
        }
        apply_normalizer_config(norm);
    }

    std::string normalize(const std::string& text) const {
        std::string value = text;
        for (const auto& step : mSteps) {
            switch (step.kind) {
            case NormalizerStep::Kind::Precompiled:
                value = precompiled_normalize(value);
                break;
            case NormalizerStep::Kind::Lowercase:
                value = lowercase_ascii(std::move(value));
                break;
            case NormalizerStep::Kind::DropChars:
                value = drop_chars(std::move(value), step.chars);
                break;
            case NormalizerStep::Kind::CollapseWhitespace:
                value = collapse_whitespace(std::move(value));
                break;
            case NormalizerStep::Kind::CollapseSpaces:
                value = collapse_spaces(std::move(value));
                break;
            case NormalizerStep::Kind::CollapseWhitespaceRuns:
                value = collapse_whitespace_runs(std::move(value));
                break;
            case NormalizerStep::Kind::Nfc:
                value = require_composed(std::move(value));
                break;
            case NormalizerStep::Kind::Strip:
                value = strip_outer_whitespace(std::move(value));
                break;
            case NormalizerStep::Kind::StripRight:
                value = strip_trailing_whitespace(std::move(value));
                break;
            }
        }
        return value;
    }

    void parse_pre_tokenizer(const nlohmann::json& j) {
        if (!j.contains("pre_tokenizer") || j["pre_tokenizer"].is_null())
            return;
        auto& pt = j["pre_tokenizer"];
        std::string ptype = pt.value("type", "");

        if (ptype == "Metaspace") {
            mAddPrefixSpace = pt.value("add_prefix_space", true);
        } else if (ptype == "Sequence") {
            // Look for Metaspace inside the sequence
            if (pt.contains("pretokenizers")) {
                for (auto& sub : pt["pretokenizers"]) {
                    if (sub.value("type", "") == "Metaspace") {
                        mAddPrefixSpace = sub.value("add_prefix_space", true);
                        break;
                    }
                }
            }
        }
    }

    void parse_added_tokens(const nlohmann::json& j) {
        if (!j.contains("added_tokens"))
            return;
        for (auto& tok : j["added_tokens"]) {
            int32_t tok_id = tok.value("id", -1);
            std::string content = tok.value("content", "");
            if (tok_id >= 0 && !content.empty()) {
                if (static_cast<size_t>(tok_id) >= mIdToToken.size()) {
                    mIdToToken.resize(static_cast<size_t>(tok_id) + 1);
                    mScores.resize(static_cast<size_t>(tok_id) + 1, 0.0f);
                }
                mIdToToken[tok_id] = content;
                mTokenToId[content] = tok_id;
            }
        }
    }

    // Extract BOS/EOS from TemplateProcessing "single" array
    void extract_template_bos_eos(const nlohmann::json& pp) {
        if (!pp.contains("single") || !pp["single"].is_array())
            return;
        bool seen_sequence = false;
        for (auto& item : pp["single"]) {
            if (item.contains("Sequence")) {
                seen_sequence = true;
                continue;
            }
            if (!item.contains("SpecialToken"))
                continue;
            std::string tok_str = item["SpecialToken"].value("id", "");
            auto it = mTokenToId.find(tok_str);
            if (it == mTokenToId.end())
                continue;
            if (!seen_sequence && mBosId < 0)
                mBosId = it->second;
            else
                mEosId = it->second;
        }
    }

    // Extract BOS/EOS from RobertaProcessing cls/sep arrays
    static int32_t extract_pp_id(const nlohmann::json& pp, const char* key) {
        if (pp.contains(key) && pp[key].is_array() && pp[key].size() >= 2)
            return pp[key][1].get<int32_t>();
        return -1;
    }

    void parse_post_processor(const nlohmann::json& j) {
        if (!j.contains("post_processor") || j["post_processor"].is_null())
            return;
        auto& pp = j["post_processor"];
        std::string ptype = pp.value("type", "");

        if (ptype == "TemplateProcessing")
            extract_template_bos_eos(pp);
        if (ptype == "RobertaProcessing") {
            mBosId = extract_pp_id(pp, "cls");
            mEosId = extract_pp_id(pp, "sep");
        }
    }

    void resolve_special_ids() {
        // Try common special-token names.
        auto find_id = [this](const std::string& a, const std::string& b) -> int32_t {
            auto it = mTokenToId.find(a);
            if (it != mTokenToId.end())
                return it->second;
            it = mTokenToId.find(b);
            return it != mTokenToId.end() ? it->second : -1;
        };

        if (mBosId < 0)
            mBosId = find_id("<s>", "[CLS]");
        if (mEosId < 0)
            mEosId = find_id("</s>", "[SEP]");
        int32_t padId = find_id("<pad>", "[PAD]");

        // Build decode skip set
        for (int32_t id : {mBosId, mEosId, padId}) {
            if (id >= 0)
                mDecodeSkipIds.insert(id);
        }
    }

    // ─── Data members ───

    std::vector<std::string> mIdToToken;
    std::vector<float> mScores;
    std::unordered_map<std::string, int32_t> mTokenToId;
    std::unordered_set<int32_t> mDecodeSkipIds;
    Trie mTrie;

    int32_t mUnkId = 0;
    float mUnkScore = -100.0f;
    bool mAddSpecialTokens = true;
    std::vector<NormalizerStep> mSteps;
    bool mAddPrefixSpace = true;

    int32_t mBosId = -1;
    int32_t mEosId = -1;
};

} // namespace

std::unique_ptr<ITokenizer> CreateDebertaTokenizer(const char* tokenizer_json_data,
                                                   std::size_t tokenizer_json_size,
                                                   bool add_special_tokens) {
    return UnigramTokenizer::Create(tokenizer_json_data, tokenizer_json_size, add_special_tokens);
}

} // namespace trtmc
