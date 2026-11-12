/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_next/runtime/tokenizer.h"

#include <algorithm>
#include <climits>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/stringpiece.h>
#include <unicode/unistr.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc::qwen3_next {
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

// ─── GPT-2 byte encoder: byte value <-> Unicode codepoint ───

struct ByteEncoderTables {
    // byte -> UTF-8 encoded string (precomputed for speed)
    std::string byte_to_utf8[256];
    // Unicode codepoint -> byte value
    std::unordered_map<char32_t, uint8_t> cp_to_byte;

    ByteEncoderTables() {
        // GPT-2 byte encoder: printable bytes map to themselves,
        // others map to 256+ to avoid control chars.
        bool direct[256] = {};
        for (int b = 33; b <= 126; ++b)
            direct[b] = true; // !"#$...~
        for (int b = 161; b <= 172; ++b)
            direct[b] = true; // non-breaking space area
        for (int b = 174; b <= 255; ++b)
            direct[b] = true; // extended latin

        int n = 0;
        for (int b = 0; b < 256; ++b) {
            char32_t cp = direct[b] ? static_cast<char32_t>(b) : static_cast<char32_t>(256 + n++);
            byte_to_utf8[b] = utf32_to_utf8(cp);
            cp_to_byte[cp] = static_cast<uint8_t>(b);
        }
    }
};

static const ByteEncoderTables& byte_tables() {
    static ByteEncoderTables tables;
    return tables;
}

// The pinned Qwen3-Next pipeline uses the pinned Unicode regex split.
class BpeTokenizer final : public ITokenizer {
  public:
    static std::unique_ptr<BpeTokenizer> Create(const char* tokenizer_json_data,
                                                std::size_t tokenizer_json_size,
                                                bool add_special_tokens = false) {
        auto tokenizer = std::unique_ptr<BpeTokenizer>(new BpeTokenizer());
        // The pinned ByteLevel post-processor adds no tokens for either flag.
        (void)add_special_tokens;
        tokenizer->parse_tokenizer_json(tokenizer_json_data, tokenizer_json_size);
        return tokenizer;
    }

    void encode_segment(const std::string& text, std::vector<int32_t>& result) const {
        encode_bytelevel(normalize_text(text), result);
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
                encode_segment(seg.text, result);
        }

        return result;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string joined = join_vocab_tokens(ids);
        return byte_decode(joined);
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
        UErrorCode status = U_ZERO_ERROR;
        const auto* normalizer = icu::Normalizer2::getNFCInstance(status);
        if (U_FAILURE(status))
            throw std::runtime_error("cannot initialize Qwen3-Next NFC");
        icu::UnicodeString result;
        normalizer->normalize(icu::UnicodeString::fromUTF8(text), result, status);
        if (U_FAILURE(status))
            throw std::runtime_error("cannot normalize Qwen3-Next input");
        std::string normalized;
        result.toUTF8String(normalized);
        return normalized;
    }

    // ─── Added token segmentation ───

    struct Segment {
        std::string text;
        int32_t added_id; /* -1 = normal */
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

    std::vector<Segment> split_added_tokens(const std::string& text) const {
        std::vector<Segment> segments;
        if (mAddedTokenPatterns.empty()) {
            segments.push_back({text, -1});
            return segments;
        }
        size_t pos = 0;
        while (pos < text.size()) {
            auto [best_id, best_len] = find_longest_added_token(text, pos);
            if (best_id >= 0) {
                segments.push_back({text.substr(pos, best_len), best_id});
                pos += best_len;
            } else {
                if (segments.empty() || segments.back().added_id >= 0) {
                    segments.push_back({"", -1});
                }
                segments.back().text.push_back(text[pos]);
                ++pos;
            }
        }
        return segments;
    }

    void encode_bytelevel(const std::string& text, std::vector<int32_t>& result) const {
        std::vector<std::string> words{text};
        for (const auto& pattern : mSplits) {
            std::vector<std::string> next;
            for (const auto& word : words) {
                const auto input = icu::UnicodeString::fromUTF8(word);
                UErrorCode status = U_ZERO_ERROR;
                std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(input, status));
                if (U_FAILURE(status))
                    throw std::runtime_error("cannot create Qwen3-Next regex matcher");
                int32_t previous = 0;
                auto emit = [&](int32_t first, int32_t last) {
                    if (first == last)
                        return;
                    std::string piece;
                    input.tempSubStringBetween(first, last).toUTF8String(piece);
                    next.push_back(std::move(piece));
                };
                while (matcher->find(status)) {
                    const auto first = matcher->start(status);
                    const auto last = matcher->end(status);
                    emit(previous, first);
                    emit(first, last);
                    previous = last;
                }
                if (U_FAILURE(status))
                    throw std::runtime_error("cannot split Qwen3-Next tokenizer input");
                emit(previous, input.length());
            }
            words = std::move(next);
        }

        for (const auto& word : words) {
            auto chars = byte_encode(word);
            auto tokens = apply_merges(std::move(chars));
            for (const auto& token : tokens) {
                auto it = mTokenToId.find(token);
                if (it != mTokenToId.end()) {
                    result.push_back(it->second);
                }
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

    static std::vector<std::string> byte_encode(const std::string& text) {
        const auto& tables = byte_tables();
        std::vector<std::string> result;
        result.reserve(text.size());
        for (unsigned char byte : text) {
            result.push_back(tables.byte_to_utf8[byte]);
        }
        return result;
    }

    static std::string byte_decode(const std::string& text) {
        const auto& tables = byte_tables();
        std::string result;
        result.reserve(text.size());
        size_t pos = 0;
        while (pos < text.size()) {
            size_t start = pos;
            char32_t cp = utf8_to_char32(text, pos);
            auto it = tables.cp_to_byte.find(cp);
            if (it != tables.cp_to_byte.end()) {
                result.push_back(static_cast<char>(it->second));
            } else {
                // Not a GPT-2 byte-encoded codepoint — pass through raw UTF-8
                result.append(text, start, pos - start);
            }
        }
        return result;
    }

    // Generic pre-tokenizer used when no GPT-2 pattern is declared.

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

    void parse_vocab(const nlohmann::json& j) {
        auto& vocab_obj = j["model"]["vocab"];
        size_t vocab_size = vocab_obj.size();
        mVocab.resize(vocab_size);

        for (auto& [token, id] : vocab_obj.items()) {
            int32_t token_id = id.get<int32_t>();
            if (token_id >= 0 && token_id < static_cast<int32_t>(vocab_size)) {
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
            std::string content = tok["content"].get<std::string>();
            int32_t id = tok["id"].get<int32_t>();
            if (content.empty() || tok.value("single_word", false) || tok.value("lstrip", false) ||
                tok.value("rstrip", false))
                throw std::runtime_error("unsupported Qwen3-Next added-token matching rule");

            if (id >= 0 && static_cast<size_t>(id) >= mVocab.size()) {
                mVocab.resize(static_cast<size_t>(id) + 1);
            }
            if (id >= 0) {
                mVocab[id] = content;
                mTokenToId[content] = id;
            }

            if (tok.value("special", false)) {
                mSpecialIds.insert(id);
            }
            // All added tokens (special and non-special) participate in pre-split
            // matching during encode, matching HuggingFace's AddedToken behavior.
            // The special flag controls decode filtering only.
            mAddedTokenPatterns.push_back({content, id});
        }
        // Sort by length descending for longest-match-first
        std::sort(mAddedTokenPatterns.begin(), mAddedTokenPatterns.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }

    void parse_tokenizer_json(const char* data, std::size_t size) {
        const auto j = nlohmann::json::parse(data, data + size);
        static const auto pretokenizer = nlohmann::json::parse(
            R"qwen({"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]})qwen");
        static const auto decoder = nlohmann::json::parse(
            R"qwen({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false})qwen");
        if (j.at("pre_tokenizer") != pretokenizer || j.at("decoder") != decoder ||
            j.at("normalizer") != nlohmann::json{{"type", "NFC"}} ||
            j.at("post_processor") != decoder)
            throw std::runtime_error("unsupported Qwen3-Next tokenizer pipeline");
        const auto& model = j.at("model");
        if (model.value("type", "") != "BPE" || !model.at("vocab").is_object() ||
            !model.at("merges").is_array() || model.value("byte_fallback", false) ||
            model.value("ignore_merges", false) || model.value("fuse_unk", false))
            throw std::runtime_error("unsupported Qwen3-Next BPE model");
        for (const auto* key : {"dropout", "unk_token"})
            if (model.contains(key) && !model.at(key).is_null())
                throw std::runtime_error("unsupported Qwen3-Next BPE model option");
        for (const auto* key : {"continuing_subword_prefix", "end_of_word_suffix"})
            if (model.contains(key) && !model.at(key).is_null() && model.at(key) != "")
                throw std::runtime_error("unsupported Qwen3-Next BPE model affix");
        for (std::size_t i = 0; i < 1; ++i) {
            const auto regex =
                pretokenizer.at("pretokenizers").at(i).at("pattern").at("Regex").get<std::string>();
            UErrorCode status = U_ZERO_ERROR;
            std::unique_ptr<icu::RegexPattern> pattern(
                icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(regex), 0, status));
            if (U_FAILURE(status) || !pattern)
                throw std::runtime_error("cannot compile Qwen3-Next tokenizer regex");
            mSplits.push_back(std::move(pattern));
        }
        parse_vocab(j);
        parse_merges(j);
        parse_added_tokens(j);
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

    std::unordered_set<int32_t> mSpecialIds;
    std::vector<std::pair<std::string, int32_t>> mAddedTokenPatterns;
    std::vector<std::unique_ptr<icu::RegexPattern>> mSplits;
};

} // namespace

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const char* data, std::size_t size,
                                               bool add_special_tokens) {
    return BpeTokenizer::Create(data, size, add_special_tokens);
}

} // namespace trtmc::qwen3_next
