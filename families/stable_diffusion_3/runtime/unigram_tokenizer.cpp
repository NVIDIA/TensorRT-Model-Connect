/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_diffusion_3/runtime/tokenizer.h"
#include "families/stable_diffusion_3/runtime/unicode.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
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

// The text without the Unicode white space at its end.
std::string trim_white_space_end(const std::string& text) {
    std::size_t keep = 0;
    std::size_t pos = 0;
    while (pos < text.size()) {
        if (!stable_diffusion_3::unicode::is_white_space(utf8_to_char32(text, pos)))
            keep = pos;
    }
    return text.substr(0, keep);
}

// The position after the Unicode white space that starts at pos.
std::size_t skip_white_space(const std::string& text, std::size_t pos) {
    while (pos < text.size()) {
        std::size_t next = pos;
        if (!stable_diffusion_3::unicode::is_white_space(utf8_to_char32(text, next)))
            break;
        pos = next;
    }
    return pos;
}

// ─── Precompiled charsmap (sentencepiece normalization rules) ───
//
// The blob is [u32 LE trie size][darts double-array trie][NUL-separated
// replacement strings]. Normalization follows the tokenizers library: each
// grapheme shorter than 6 bytes is looked up whole, otherwise (or when it has
// no rule) each code point is looked up on its own; the first rule found on
// the key wins. Grapheme clusters follow UAX #29 (generated tables) for what
// decides a lookup here: control characters, CR and LF stand alone (CR LF
// together), prepended characters join what follows, and a code point takes
// the Extend, ZWJ and SpacingMark code points after it. Hangul syllable and
// regional-indicator sequences are not joined; such clusters are at least 6
// bytes long, so they are looked up code point by code point either way. Emoji
// ZWJ (GB11) and Indic conjunct (GB9c) sequences are not joined either, which
// differs from the library when the cluster left after the joiner is shorter
// than 6 bytes and starts with a code point that has a rule (for example an
// emoji, ZWJ, U+2122 and an accent: here the accent is dropped). The
// darts-clone double-array layout is read from its published format.

std::string decode_base64(const std::string& text) {
    auto value = [](char c) -> int {
        if (c >= 'A' && c <= 'Z')
            return c - 'A';
        if (c >= 'a' && c <= 'z')
            return c - 'a' + 26;
        if (c >= '0' && c <= '9')
            return c - '0' + 52;
        if (c == '+')
            return 62;
        if (c == '/')
            return 63;
        return -1;
    };
    std::string out;
    out.reserve(text.size() * 3 / 4);
    std::uint32_t buffer = 0;
    int bits = 0;
    for (char c : text) {
        if (c == '=')
            break;
        const int v = value(c);
        if (v < 0)
            throw std::runtime_error("precompiled_charsmap is not valid base64");
        buffer = (buffer << 6) | static_cast<std::uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((buffer >> bits) & 0xFF));
        }
    }
    return out;
}

class PrecompiledCharsmap {
  public:
    explicit PrecompiledCharsmap(const std::string& blob) {
        if (blob.size() < 4)
            throw std::runtime_error("precompiled_charsmap is truncated");
        std::uint32_t size = 0;
        for (int i = 3; i >= 0; --i)
            size = (size << 8) | static_cast<unsigned char>(blob[static_cast<std::size_t>(i)]);
        if (size % 4U != 0U || 4U + static_cast<std::size_t>(size) > blob.size())
            throw std::runtime_error("precompiled_charsmap trie size is inconsistent");
        units_.resize(size / 4U);
        for (std::size_t unit = 0; unit < units_.size(); ++unit) {
            std::uint32_t word = 0;
            for (int i = 3; i >= 0; --i)
                word = (word << 8) | static_cast<unsigned char>(
                                         blob[4U + unit * 4U + static_cast<std::size_t>(i)]);
            units_[unit] = word;
        }
        normalized_ = blob.substr(4U + size);
    }

    // The replacement of the first rule that matches a prefix of key, if any.
    bool transform(const char* key, std::size_t length, std::string& replacement) const {
        if (units_.empty())
            return false;
        std::size_t node = offset(units_[0]);
        for (std::size_t i = 0; i < length; ++i) {
            const auto c = static_cast<unsigned char>(key[i]);
            if (c == 0)
                break;
            node ^= c;
            if (node >= units_.size())
                return false;
            const std::uint32_t unit = units_[node];
            if (label(unit) != c)
                return false;
            node ^= offset(unit);
            if (has_leaf(unit)) {
                if (node >= units_.size())
                    return false;
                const std::size_t start = value(units_[node]);
                if (start >= normalized_.size())
                    return false;
                const std::size_t stop = normalized_.find('\0', start);
                replacement = normalized_.substr(
                    start, (stop == std::string::npos ? normalized_.size() : stop) - start);
                return true;
            }
        }
        return false;
    }

    std::string normalize(const std::string& text) const {
        std::string result;
        result.reserve(text.size());
        std::string replacement;
        std::size_t pos = 0;
        while (pos < text.size()) {
            const std::size_t start = pos;
            pos = grapheme_end(text, pos);
            if (pos - start < 6 && transform(text.data() + start, pos - start, replacement)) {
                result += replacement;
                continue;
            }
            std::size_t part = start;
            while (part < pos) {
                std::size_t next = part;
                (void)utf8_to_char32(text, next);
                if (transform(text.data() + part, next - part, replacement))
                    result += replacement;
                else
                    result.append(text, part, next - part);
                part = next;
            }
        }
        return result;
    }

  private:
    // End of the grapheme cluster that starts at pos.
    static std::size_t grapheme_end(const std::string& text, std::size_t pos) {
        namespace unicode = stable_diffusion_3::unicode;
        char32_t current = utf8_to_char32(text, pos);
        if (unicode::is_grapheme_control(current)) {
            if (current == '\r' && pos < text.size() && text[pos] == '\n')
                ++pos;
            return pos;
        }
        while (unicode::is_grapheme_prepend(current) && pos < text.size()) {
            std::size_t probe = pos;
            const char32_t next = utf8_to_char32(text, probe);
            if (unicode::is_grapheme_control(next))
                return pos;
            current = next;
            pos = probe;
        }
        while (pos < text.size()) {
            std::size_t probe = pos;
            if (!unicode::is_grapheme_extend(utf8_to_char32(text, probe)))
                break;
            pos = probe;
        }
        return pos;
    }
    static bool has_leaf(std::uint32_t unit) { return ((unit >> 8) & 1U) == 1U; }
    static std::uint32_t value(std::uint32_t unit) { return unit & ((1U << 31) - 1U); }
    static std::uint32_t label(std::uint32_t unit) { return unit & ((1U << 31) | 0xFFU); }
    static std::uint32_t offset(std::uint32_t unit) {
        return (unit >> 10) << ((unit & (1U << 9)) >> 6);
    }

    std::vector<std::uint32_t> units_;
    std::string normalized_;
};

// ─── Metaspace Pre-tokenizer ───
// Replaces spaces with ▁ (U+2581) and optionally adds prefix space.

static const std::string kMetaspaceChar = "\xe2\x96\x81"; // ▁ U+2581

// ─── Trie for efficient vocab prefix lookup ───

struct TrieNode {
    std::unordered_map<char, int> children;
    int token_id = -1; // -1 = not a token end
    double score = 0.0;
};

class Trie {
  public:
    Trie() { mNodes.emplace_back(); } // root node

    void insert(const std::string& token, int id, double score) {
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
        double score;
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
// Scores are doubles and ties keep the first candidate, as in the tokenizers
// library: near-equal paths are otherwise decided by float rounding.

struct ViterbiNode {
    double score;
    int token_id;
    size_t prev_pos; // byte position of previous node
};

std::vector<int32_t> viterbi_encode(const std::string& text, const Trie& trie, int32_t unk_id,
                                    double unk_score) {
    if (text.empty())
        return {};

    size_t n = text.size();
    // best[i] = best path score to reach byte position i
    std::vector<ViterbiNode> best(n + 1, {-std::numeric_limits<double>::infinity(), -1, 0});
    best[0].score = 0.0;

    std::vector<Trie::Match> matches;

    for (size_t i = 0; i < n; ++i) {
        if (best[i].score == -std::numeric_limits<double>::infinity())
            continue;

        trie.find_prefixes(text, i, matches);

        for (const auto& m : matches) {
            size_t end = i + m.length;
            double new_score = best[i].score + m.score;
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
            double new_score = best[i].score + unk_score;
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

        // Special added tokens are matched in the raw text first, as the
        // tokenizers library does (leftmost, then longest; lstrip / rstrip
        // absorb the white space next to them); the text between them is
        // tokenized normally.
        std::vector<int32_t> ids;
        std::size_t start = 0;
        while (start < text.size()) {
            std::size_t found = std::string::npos;
            const SpecialToken* match = nullptr;
            for (const auto& token : mSpecialTokens) {
                const std::size_t at = text.find(token.content, start);
                if (at != std::string::npos &&
                    (at < found || (at == found && token.content.size() > match->content.size()))) {
                    found = at;
                    match = &token;
                }
            }
            std::string plain =
                text.substr(start, found == std::string::npos ? std::string::npos : found - start);
            if (match != nullptr && match->lstrip)
                plain = trim_white_space_end(plain);
            encode_plain(plain, start == 0, ids);
            if (match == nullptr)
                break;
            ids.push_back(match->id);
            start = found + match->content.size();
            if (match->rstrip)
                start = skip_white_space(text, start);
        }

        if (mAddSpecialTokens)
            ids = make_special_frame(ids);
        return ids;
    }

    // The text between special tokens: the tokenizer.json normalizer
    // sequence, then its pre-tokenizer, then Unigram per piece.
    void encode_plain(const std::string& text, bool at_start, std::vector<int32_t>& ids) const {
        std::string normalized = text;
        for (const auto& step : mNormalizers)
            normalized = apply(step, normalized);

        std::vector<std::string> pieces;
        if (mWhitespaceSplit) {
            pieces = unicode_whitespace_split(normalized);
        } else if (!normalized.empty()) {
            pieces.push_back(normalized);
        }
        std::vector<std::string> words;
        for (std::size_t i = 0; i < pieces.size(); ++i) {
            std::string piece = replace_all(pieces[i], " ", mReplacement);
            const bool prepend =
                mPrepend == Prepend::kAlways || (mPrepend == Prepend::kFirst && at_start && i == 0);
            if (prepend && piece.compare(0, mReplacement.size(), mReplacement) != 0)
                piece = mReplacement + piece;
            if (!mSplit) {
                words.push_back(std::move(piece));
                continue;
            }
            // Split on the replacement; each delimiter stays with the text after it.
            std::size_t from = 0;
            while (from < piece.size()) {
                std::size_t search = from;
                if (piece.compare(from, mReplacement.size(), mReplacement) == 0)
                    search += mReplacement.size();
                std::size_t next = piece.find(mReplacement, search);
                if (next == std::string::npos)
                    next = piece.size();
                words.push_back(piece.substr(from, next - from));
                from = next;
            }
        }
        for (const auto& word : words) {
            auto word_ids = viterbi_encode(word, mTrie, mUnkId, mUnkScore);
            // Consecutive unknown pieces of one word fuse into one, like the
            // tokenizers library's Unigram model.
            for (std::size_t k = 0; k < word_ids.size(); ++k) {
                if (k > 0 && word_ids[k] == mUnkId && word_ids[k - 1] == mUnkId)
                    continue;
                ids.push_back(word_ids[k]);
            }
        }
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
            double score = entry[1].get<double>();

            mIdToToken[i] = token;
            mTokenToId[token] = static_cast<int32_t>(i);
            mScores.push_back(score);
        }

        // UNK score: the lowest vocab score minus 10, as the tokenizers library sets it.
        double min_score = std::numeric_limits<double>::infinity();
        for (double s : mScores) {
            if (s < min_score)
                min_score = s;
        }
        mUnkScore = min_score - 10.0;
    }

    void build_trie() {
        for (size_t i = 0; i < mIdToToken.size(); ++i) {
            const auto& token = mIdToToken[i];
            if (!token.empty()) {
                mTrie.insert(token, static_cast<int>(i), mScores[i]);
            }
        }
    }

    struct NormalizerStep {
        enum class Kind { kPrecompiled, kStrip, kReplace, kLowercase };
        Kind kind{Kind::kPrecompiled};
        bool left{false};
        bool right{false};
        bool space_runs{false}; // the regex " {2,}"
        std::string pattern;
        std::string content;
    };

    // Normalizers this tokenizer can reproduce; anything else is refused.
    void apply_normalizer_config(const nlohmann::json& norm) {
        const std::string type = norm.value("type", "");
        NormalizerStep step;
        if (type == "Precompiled") {
            const auto charsmap = norm.value("precompiled_charsmap", std::string{});
            if (charsmap.empty())
                throw std::runtime_error("tokenizer.json Precompiled normalizer has no charsmap");
            mCharsmap = std::make_unique<PrecompiledCharsmap>(decode_base64(charsmap));
            step.kind = NormalizerStep::Kind::kPrecompiled;
        } else if (type == "Strip") {
            step.kind = NormalizerStep::Kind::kStrip;
            step.left = norm.value("strip_left", false);
            step.right = norm.value("strip_right", false);
        } else if (type == "Replace") {
            step.kind = NormalizerStep::Kind::kReplace;
            step.content = norm.value("content", std::string{});
            const auto& pattern = norm.at("pattern");
            if (pattern.contains("String")) {
                step.pattern = pattern["String"].get<std::string>();
                if (step.pattern.empty())
                    throw std::runtime_error("tokenizer.json Replace has an empty pattern");
            } else if (pattern.value("Regex", std::string{}) == " {2,}") {
                step.space_runs = true;
            } else {
                throw std::runtime_error("tokenizer.json Replace pattern is not supported: " +
                                         pattern.dump());
            }
        } else if (type == "Lowercase") {
            step.kind = NormalizerStep::Kind::kLowercase;
        } else {
            throw std::runtime_error("tokenizer.json normalizer is not supported: " + type);
        }
        mNormalizers.push_back(std::move(step));
    }

    void parse_normalizer(const nlohmann::json& j) {
        if (!j.contains("normalizer") || j["normalizer"].is_null())
            return;
        const auto& norm = j["normalizer"];
        if (norm.value("type", "") == "Sequence") {
            for (const auto& sub : norm.at("normalizers"))
                apply_normalizer_config(sub);
            return;
        }
        apply_normalizer_config(norm);
    }

    void apply_pre_tokenizer_config(const nlohmann::json& pt) {
        const std::string type = pt.value("type", "");
        if (type == "WhitespaceSplit" && !mMetaspace) {
            mWhitespaceSplit = true;
            return;
        }
        if (type != "Metaspace" || mMetaspace)
            throw std::runtime_error("tokenizer.json pre_tokenizer is not supported: " + pt.dump());
        mMetaspace = true;
        mReplacement = pt.value("replacement", kMetaspaceChar);
        mSplit = pt.value("split", true);
        if (!pt.contains("prepend_scheme")) {
            mPrepend = pt.value("add_prefix_space", true) ? Prepend::kAlways : Prepend::kNever;
            return;
        }
        const auto scheme = pt["prepend_scheme"].get<std::string>();
        if (scheme == "always")
            mPrepend = Prepend::kAlways;
        else if (scheme == "first")
            mPrepend = Prepend::kFirst;
        else if (scheme == "never")
            mPrepend = Prepend::kNever;
        else
            throw std::runtime_error("tokenizer.json Metaspace prepend_scheme: " + scheme);
    }

    void parse_pre_tokenizer(const nlohmann::json& j) {
        if (!j.contains("pre_tokenizer") || j["pre_tokenizer"].is_null())
            throw std::runtime_error("tokenizer.json has no pre_tokenizer");
        const auto& pt = j["pre_tokenizer"];
        if (pt.value("type", "") == "Sequence") {
            for (const auto& sub : pt.at("pretokenizers"))
                apply_pre_tokenizer_config(sub);
        } else {
            apply_pre_tokenizer_config(pt);
        }
        if (!mMetaspace)
            throw std::runtime_error("tokenizer.json pre_tokenizer has no Metaspace step");
    }

    std::string apply(const NormalizerStep& step, const std::string& text) const {
        switch (step.kind) {
        case NormalizerStep::Kind::kPrecompiled:
            return mCharsmap->normalize(text);
        case NormalizerStep::Kind::kStrip:
            return strip_whitespace(text, step.left, step.right);
        case NormalizerStep::Kind::kReplace:
            return step.space_runs ? replace_space_runs(text, step.content)
                                   : replace_all(text, step.pattern, step.content);
        case NormalizerStep::Kind::kLowercase: {
            std::u32string lowered;
            for (const char32_t cp : stable_diffusion_3::unicode::decode_utf8(text)) {
                if (cp == 0x0130) {
                    lowered.push_back(U'i');
                    lowered.push_back(0x0307);
                } else {
                    lowered.push_back(stable_diffusion_3::unicode::lowercase(cp));
                }
            }
            return stable_diffusion_3::unicode::encode_utf8(lowered);
        }
        }
        return text;
    }

    // Unicode White_Space, as the tokenizers library's strip and split use it.
    static bool unicode_whitespace(char32_t cp) {
        return cp == ' ' || (cp >= 0x09 && cp <= 0x0D) || cp == 0x85 || cp == 0xA0 ||
               cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
               cp == 0x202F || cp == 0x205F || cp == 0x3000;
    }

    static std::string strip_whitespace(const std::string& text, bool left, bool right) {
        const std::u32string codepoints = stable_diffusion_3::unicode::decode_utf8(text);
        std::size_t begin = 0;
        std::size_t end = codepoints.size();
        while (left && begin < end && unicode_whitespace(codepoints[begin]))
            ++begin;
        while (right && end > begin && unicode_whitespace(codepoints[end - 1]))
            --end;
        return stable_diffusion_3::unicode::encode_utf8(codepoints.substr(begin, end - begin));
    }

    static std::string replace_all(const std::string& text, const std::string& pattern,
                                   const std::string& content) {
        std::string result;
        std::size_t from = 0;
        for (std::size_t at = text.find(pattern); at != std::string::npos;
             at = text.find(pattern, from)) {
            result.append(text, from, at - from);
            result += content;
            from = at + pattern.size();
        }
        result.append(text, from, std::string::npos);
        return result;
    }

    static std::string replace_space_runs(const std::string& text, const std::string& content) {
        std::string result;
        std::size_t i = 0;
        while (i < text.size()) {
            if (text[i] == ' ' && i + 1 < text.size() && text[i + 1] == ' ') {
                while (i < text.size() && text[i] == ' ')
                    ++i;
                result += content;
                continue;
            }
            result.push_back(text[i++]);
        }
        return result;
    }

    static std::vector<std::string> unicode_whitespace_split(const std::string& text) {
        std::vector<std::string> words;
        std::u32string word;
        for (const char32_t cp : stable_diffusion_3::unicode::decode_utf8(text)) {
            if (!unicode_whitespace(cp)) {
                word.push_back(cp);
                continue;
            }
            if (!word.empty())
                words.push_back(stable_diffusion_3::unicode::encode_utf8(word));
            word.clear();
        }
        if (!word.empty())
            words.push_back(stable_diffusion_3::unicode::encode_utf8(word));
        return words;
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
                    mScores.resize(static_cast<size_t>(tok_id) + 1, 0.0);
                }
                mIdToToken[tok_id] = content;
                mTokenToId[content] = tok_id;
                if (tok.value("special", false) && !tok.value("normalized", true))
                    mSpecialTokens.push_back(
                        {content, tok_id, tok.value("lstrip", false), tok.value("rstrip", false)});
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
    std::vector<double> mScores;
    std::unordered_map<std::string, int32_t> mTokenToId;
    std::unordered_set<int32_t> mDecodeSkipIds;
    Trie mTrie;

    int32_t mUnkId = 0;
    double mUnkScore = -100.0;
    bool mAddSpecialTokens = true;
    std::unique_ptr<PrecompiledCharsmap> mCharsmap;
    std::vector<NormalizerStep> mNormalizers;
    enum class Prepend { kAlways, kFirst, kNever };
    bool mWhitespaceSplit = false;
    bool mMetaspace = false;
    bool mSplit = true;
    Prepend mPrepend = Prepend::kAlways;
    std::string mReplacement = kMetaspaceChar;
    struct SpecialToken {
        std::string content;
        int32_t id;
        bool lstrip; // absorbs the white space before it
        bool rstrip; // absorbs the white space after it
    };
    std::vector<SpecialToken> mSpecialTokens;

    int32_t mBosId = -1;
    int32_t mEosId = -1;
};

} // namespace

std::unique_ptr<ITokenizer> CreateUnigramTokenizer(const char* tokenizer_json_data,
                                                   std::size_t tokenizer_json_size,
                                                   bool add_special_tokens) {
    return UnigramTokenizer::Create(tokenizer_json_data, tokenizer_json_size, add_special_tokens);
}

} // namespace trtmc
