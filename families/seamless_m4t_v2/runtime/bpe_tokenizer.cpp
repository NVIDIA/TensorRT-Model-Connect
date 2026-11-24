/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/seamless_m4t_v2/runtime/tokenizer.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>
#include <list>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <string>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

// ─── UTF-8 helpers ───

// Decode one UTF-8 codepoint from s starting at pos, advance pos.
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

inline std::string utf32_to_utf8(char32_t cp) {
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

// Read one UTF-8 codepoint from raw bytes, advance ptr. Returns 0xFFFD on error.
inline char32_t read_utf8(const char*& p, const char* end) {
    if (p >= end)
        return 0xFFFD;
    unsigned char c = static_cast<unsigned char>(*p);
    if (c < 0x80) {
        ++p;
        return c;
    }
    if ((c & 0xE0) == 0xC0 && p + 1 < end) {
        char32_t cp =
            (static_cast<char32_t>(c & 0x1F) << 6) | (static_cast<unsigned char>(p[1]) & 0x3F);
        p += 2;
        return cp;
    }
    if ((c & 0xF0) == 0xE0 && p + 2 < end) {
        char32_t cp = (static_cast<char32_t>(c & 0x0F) << 12) |
                      (static_cast<char32_t>(static_cast<unsigned char>(p[1]) & 0x3F) << 6) |
                      (static_cast<unsigned char>(p[2]) & 0x3F);
        p += 3;
        return cp;
    }
    if ((c & 0xF8) == 0xF0 && p + 3 < end) {
        char32_t cp = (static_cast<char32_t>(c & 0x07) << 18) |
                      (static_cast<char32_t>(static_cast<unsigned char>(p[1]) & 0x3F) << 12) |
                      (static_cast<char32_t>(static_cast<unsigned char>(p[2]) & 0x3F) << 6) |
                      (static_cast<unsigned char>(p[3]) & 0x3F);
        p += 4;
        return cp;
    }
    ++p;
    return 0xFFFD;
}

class BpeTokenizer final : public ITokenizer {
  public:
    static std::unique_ptr<BpeTokenizer> Create(const char* tokenizer_json_data,
                                                std::size_t tokenizer_json_size,
                                                bool add_special_tokens,
                                                std::size_t vocabulary_size) {
        auto tokenizer = std::unique_ptr<BpeTokenizer>(new BpeTokenizer());
        if (add_special_tokens || vocabulary_size == 0)
            throw std::invalid_argument("SeamlessM4T inserts language/EOS tokens in the runtime; "
                                        "a positive model vocabulary bound is required");
        tokenizer->mVocabularySize = vocabulary_size;
        tokenizer->parse_tokenizer_json(tokenizer_json_data, tokenizer_json_size);
        return tokenizer;
    }

    std::vector<int32_t> encode(const std::string& text) const override {
        std::vector<int32_t> result;
        if (text.empty())
            return result;

        auto segments = split_added_tokens(text);
        for (const auto& seg : segments) {
            if (seg.added_id >= 0)
                result.push_back(seg.added_id);
            else
                encode_sentencepiece(normalize_text(seg.text), result, seg.offset == 0);
        }

        return result;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string joined = join_vocab_tokens(ids);
        return decode_metaspace(joined);
    }

    int32_t id_for_token(std::string_view token) const override {
        auto it = mTokenToId.find(std::string(token));
        return it != mTokenToId.end() ? it->second : -1;
    }

    std::string token_for_id(int32_t id) const override {
        if (id >= 0 && static_cast<size_t>(id) < mVocab.size()) {
            return mVocab[id];
        }
        return "";
    }

  private:
    BpeTokenizer() = default;

    std::string normalize_text(const std::string& text) const {
        auto normalized = icu::UnicodeString::fromUTF8(text);
        for (const auto& operation : mTextNormalizer.at("normalizers")) {
            UErrorCode status = U_ZERO_ERROR;
            const std::string type = operation.at("type");
            if (type == "NFKC") {
                auto* normalizer = icu::Normalizer2::getNFKCInstance(status);
                icu::UnicodeString next;
                if (U_SUCCESS(status))
                    normalizer->normalize(normalized, next, status);
                normalized = std::move(next);
            } else if (type == "Strip") {
                if (operation.value("strip_left", false)) {
                    int32_t first = 0;
                    while (first < normalized.length() &&
                           u_isUWhiteSpace(normalized.char32At(first)))
                        first = normalized.moveIndex32(first, 1);
                    normalized.remove(0, first);
                }
                if (operation.value("strip_right", false)) {
                    int32_t end = normalized.length();
                    while (end > 0) {
                        const int32_t previous = normalized.moveIndex32(end, -1);
                        if (!u_isUWhiteSpace(normalized.char32At(previous)))
                            break;
                        end = previous;
                    }
                    normalized.truncate(end);
                }
            } else if (type == "Replace") {
                const auto replacement =
                    icu::UnicodeString::fromUTF8(operation.at("content").get<std::string>());
                const auto& pattern = operation.at("pattern");
                if (pattern.contains("Regex")) {
                    std::unique_ptr<icu::RegexPattern> regex(icu::RegexPattern::compile(
                        icu::UnicodeString::fromUTF8(pattern.at("Regex").get<std::string>()), 0,
                        status));
                    if (U_FAILURE(status))
                        throw std::runtime_error("Invalid SeamlessM4T normalization regex");
                    std::unique_ptr<icu::RegexMatcher> matcher(regex->matcher(normalized, status));
                    if (U_FAILURE(status))
                        throw std::runtime_error("Invalid SeamlessM4T normalization matcher");
                    normalized = matcher->replaceAll(replacement, status);
                } else {
                    normalized.findAndReplace(
                        icu::UnicodeString::fromUTF8(pattern.at("String").get<std::string>()),
                        replacement);
                }
            } else
                throw std::runtime_error("Unsupported SeamlessM4T text normalizer");
            if (U_FAILURE(status))
                throw std::runtime_error("SeamlessM4T normalization failed");
        }
        std::string result;
        normalized.toUTF8String(result);
        return result;
    }

    struct Segment {
        std::string text;
        int32_t added_id; /* -1 = normal */
        std::size_t offset;
    };

    std::pair<int32_t, size_t> find_longest_added_token(const std::string& text, size_t pos) const {
        int32_t best_id = -1;
        size_t best_len = 0;
        for (const auto& [content, id] : mAddedTokenPatterns) {
            if (pos + content.size() <= text.size() && content.size() > best_len &&
                text.compare(pos, content.size(), content) == 0) {
                best_id = id;
                best_len = content.size();
            }
        }
        return {best_id, best_len};
    }

    static void strip_added_token_left_space(std::string& text) {
        const char* begin = text.data();
        const char* cursor = begin;
        const char* end = begin + text.size();
        size_t content_end = 0;
        while (cursor < end) {
            const char32_t cp = read_utf8(cursor, end);
            // AddedToken lstrip uses the Unicode White_Space property.
            const bool whitespace = (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 ||
                                    cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
                                    cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
                                    cp == 0x3000;
            if (!whitespace)
                content_end = static_cast<size_t>(cursor - begin);
        }
        text.resize(content_end);
    }

    std::vector<Segment> split_added_tokens(const std::string& text) const {
        std::vector<Segment> segments;
        if (mAddedTokenPatterns.empty()) {
            segments.push_back({text, -1, 0});
            return segments;
        }
        size_t pos = 0;
        while (pos < text.size()) {
            auto [best_id, best_len] = find_longest_added_token(text, pos);
            if (best_id >= 0) {
                if (mLstripAddedTokenIds.count(best_id) && !segments.empty() &&
                    segments.back().added_id < 0) {
                    strip_added_token_left_space(segments.back().text);
                }
                segments.push_back({text.substr(pos, best_len), best_id, pos});
                pos += best_len;
            } else {
                if (segments.empty() || segments.back().added_id >= 0) {
                    segments.push_back({"", -1, pos});
                }
                segments.back().text.push_back(text[pos]);
                ++pos;
            }
        }
        return segments;
    }

    // ─── Encoding helpers ───

    std::string normalize_sentencepiece(const std::string& text, bool first) const {
        static const std::string sp = "\xe2\x96\x81"; // U+2581
        std::string out;
        for (char c : text) {
            out += (c == ' ') ? sp : std::string(1, c);
        }
        // Metaspace prepend_scheme=first: prepend if not already starting with ▁
        if (first && (out.empty() || out.compare(0, sp.size(), sp) != 0)) {
            out = sp + out;
        }
        return out;
    }

    void encode_sentencepiece(const std::string& text, std::vector<int32_t>& result,
                              bool first) const {
        std::string normalized = normalize_sentencepiece(text, first);

        // Split into UTF-8 characters
        std::vector<std::string> chars;
        const char* cp = normalized.data();
        const char* ce = cp + normalized.size();
        while (cp < ce) {
            const char* cs = cp;
            read_utf8(cp, ce);
            chars.emplace_back(cs, cp);
        }

        // BPE merge and lookup
        auto tokens = apply_merges(std::move(chars));
        bool previous_unknown = false;
        for (const auto& token : tokens) {
            auto it = mTokenToId.find(token);
            if (it != mTokenToId.end()) {
                result.push_back(it->second);
                previous_unknown = false;
            } else {
                if (!previous_unknown)
                    result.push_back(mTokenToId.at("<unk>"));
                previous_unknown = true;
            }
        }
    }

    std::string join_vocab_tokens(const std::vector<int32_t>& ids) const {
        std::string joined;
        for (int32_t id : ids) {
            if (mSpecialIds.count(id))
                continue;
            if (id >= 0 && static_cast<size_t>(id) < mVocab.size()) {
                joined += mVocab[id];
            }
        }
        return joined;
    }

    std::string decode_metaspace(const std::string& joined) const {
        std::string result;
        const std::string g_char = utf32_to_utf8(0x0120);
        const std::string spiece_char = "\xe2\x96\x81"; // U+2581
        size_t pos = 0;
        while (pos < joined.size()) {
            if (pos + g_char.size() <= joined.size() &&
                joined.compare(pos, g_char.size(), g_char) == 0) {
                result.push_back(' ');
                pos += g_char.size();
            } else if (pos + spiece_char.size() <= joined.size() &&
                       joined.compare(pos, spiece_char.size(), spiece_char) == 0) {
                result.push_back(' ');
                pos += spiece_char.size();
            } else {
                result.push_back(joined[pos]);
                ++pos;
            }
        }
        if (!result.empty() && result[0] == ' ') {
            result.erase(0, 1);
        }
        return result;
    }

    // ─── BPE merge algorithm ───

    struct MergeCandidate {
        int rank;
        std::string first;
        std::string second;
    };

    MergeCandidate find_best_merge(const std::vector<std::string>& tokens) const {
        MergeCandidate best{INT_MAX, "", ""};
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            auto it =
                mMergeRank.find(std::make_pair(std::cref(tokens[i]), std::cref(tokens[i + 1])));
            if (it != mMergeRank.end() && it->second < best.rank) {
                best = {it->second, tokens[i], tokens[i + 1]};
            }
        }
        return best;
    }

    static std::vector<std::string> merge_all_pairs(std::vector<std::string> tokens,
                                                    const std::string& first,
                                                    const std::string& second) {
        std::string merged = first + second;
        std::vector<std::string> result;
        result.reserve(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i + 1 < tokens.size() && tokens[i] == first && tokens[i + 1] == second) {
                result.push_back(merged);
                ++i; // skip next
            } else {
                result.push_back(std::move(tokens[i]));
            }
        }
        return result;
    }

    // Optimized: merge ALL occurrences of the best pair per pass.
    std::vector<std::string> apply_merges(std::vector<std::string> tokens) const {
        if (tokens.size() <= 1)
            return tokens;

        while (true) {
            auto best = find_best_merge(tokens);
            if (best.rank == INT_MAX)
                break;
            tokens = merge_all_pairs(std::move(tokens), best.first, best.second);
            if (tokens.size() <= 1)
                break;
        }

        return tokens;
    }

    // ─── JSON parsing helpers ───

    static bool is_eos_content(const std::string& s) {
        return s == "<|endoftext|>" || s == "</s>" || s == "<|end_of_text|>";
    }

    void parse_vocab(const nlohmann::json& j) {
        auto& vocab_obj = j["model"]["vocab"];
        size_t vocab_size = vocab_obj.size();
        if (vocab_size > mVocabularySize)
            throw std::runtime_error("SeamlessM4T tokenizer exceeds model vocabulary");
        mVocab.resize(vocab_size);

        for (auto& [token, id] : vocab_obj.items()) {
            const auto raw_id = id.get<int64_t>();
            if (raw_id < 0 || static_cast<std::size_t>(raw_id) >= vocab_size)
                throw std::runtime_error("SeamlessM4T base token ID is outside vocabulary");
            const auto token_id = static_cast<int32_t>(raw_id);
            if (token_id >= 0) {
                mVocab[token_id] = token;
                mTokenToId[token] = token_id;
            }
        }
    }

    void parse_merges(const nlohmann::json& j) {
        if (!j["model"].contains("merges"))
            throw std::runtime_error("Invalid tokenizer.json: missing model.merges");

        auto& merges_arr = j["model"]["merges"];
        mMergeRank.reserve(merges_arr.size());

        for (size_t i = 0; i < merges_arr.size(); ++i) {
            std::string first, second;

            if (merges_arr[i].is_array()) {
                auto arr = merges_arr[i].get<std::vector<std::string>>();
                if (arr.size() != 2)
                    continue;
                first = std::move(arr[0]);
                second = std::move(arr[1]);
            } else if (merges_arr[i].is_string()) {
                std::string merge_str = merges_arr[i].get<std::string>();
                auto space_pos = merge_str.find(' ');
                if (space_pos == std::string::npos)
                    continue;
                first = merge_str.substr(0, space_pos);
                second = merge_str.substr(space_pos + 1);
            } else {
                continue;
            }

            auto pair = std::make_pair(std::move(first), std::move(second));
            mMergeRank[pair] = static_cast<int>(i);
        }
    }

    void parse_added_tokens(const nlohmann::json& j) {
        if (!j.contains("added_tokens"))
            return;
        for (auto& tok : j["added_tokens"]) {
            std::string content = tok.at("content").get<std::string>();
            const auto raw_id = tok.at("id").get<int64_t>();
            if (raw_id < 0 || static_cast<std::size_t>(raw_id) >= mVocabularySize)
                throw std::runtime_error("SeamlessM4T added token ID is outside model vocabulary");
            const auto id = static_cast<int32_t>(raw_id);
            if (static_cast<size_t>(id) >= mVocab.size()) {
                mVocab.resize(static_cast<size_t>(id) + 1);
            }
            if (id >= 0) {
                mVocab[id] = content;
                mTokenToId[content] = id;
            }

            if (tok.value("special", false)) {
                mSpecialTokens[content] = id;
                mSpecialIds.insert(id);
            }
            // All added tokens (special and non-special) participate in pre-split
            // matching during encode, matching HuggingFace's AddedToken behavior.
            // The special flag only controls decode filtering (mSpecialIds) and
            // post_processor BOS/EOS insertion.
            mAddedTokenPatterns.push_back({content, id});
            if (tok.value("lstrip", false))
                mLstripAddedTokenIds.insert(id);
        }
        // Sort by length descending for longest-match-first
        std::sort(mAddedTokenPatterns.begin(), mAddedTokenPatterns.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }

    static std::string optional_model_string(const nlohmann::json& model, const char* key) {
        auto it = model.find(key);
        if (it != model.end() && it->is_string())
            return it->get<std::string>();
        return {};
    }

    void parse_tokenizer_json(const char* json_data, std::size_t json_size) {
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(json_data, json_data + json_size);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse tokenizer.json: ") + e.what());
        }

        if (!j.contains("model"))
            throw std::runtime_error("Invalid tokenizer.json: missing model");
        auto& model = j["model"];
        if (!model.contains("vocab") || !model["vocab"].is_object())
            throw std::runtime_error("Invalid tokenizer.json: model.vocab must be an object");
        if (!model.contains("merges") || !model["merges"].is_array())
            throw std::runtime_error("Invalid tokenizer.json: model.merges must be an array");

        if (model.value("type", "") != "BPE" || model.value("byte_fallback", false) ||
            !optional_model_string(model, "end_of_word_suffix").empty())
            throw std::runtime_error("Unsupported SeamlessM4T BPE model options");
        parse_vocab(j);
        parse_merges(j);
        parse_added_tokens(j);
        mTextNormalizer = j.at("normalizer");
        if (mTextNormalizer.value("type", "") != "Sequence")
            throw std::runtime_error("Expected SeamlessM4T normalizer sequence");
        // The builder exports Metaspace; language prefixes and EOS are runtime-owned.
        for (const auto* key : {"pre_tokenizer", "decoder"}) {
            const auto& part = j.at(key);
            if (part.value("type", "") != "Metaspace" ||
                part.value("replacement", "") != "\xe2\x96\x81" ||
                part.value("prepend_scheme", "") != "first" || !part.value("split", true))
                throw std::runtime_error(std::string("Unsupported SeamlessM4T ") + key);
        }
    }

    // ─── Data members ───

    std::vector<std::string> mVocab;
    std::unordered_map<std::string, int32_t> mTokenToId;

    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            size_t h1 = std::hash<std::string>{}(p.first);
            size_t h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };

    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> mMergeRank;

    std::unordered_map<std::string, int32_t> mSpecialTokens;
    std::unordered_set<int32_t> mSpecialIds; // O(1) lookup in decode()
    int32_t mEosId = -1;

    // Non-special added tokens: matched before pre-tokenization (longest first)
    std::vector<std::pair<std::string, int32_t>> mAddedTokenPatterns;
    std::unordered_set<int32_t> mLstripAddedTokenIds;

    std::size_t mVocabularySize = 0;
    nlohmann::json mTextNormalizer;
};

} // namespace

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const char* tokenizer_json_data,
                                               std::size_t tokenizer_json_size,
                                               bool add_special_tokens,
                                               std::size_t vocabulary_size) {
    return BpeTokenizer::Create(tokenizer_json_data, tokenizer_json_size, add_special_tokens,
                                vocabulary_size);
}

} // namespace trtmc
