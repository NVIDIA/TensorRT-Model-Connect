/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trtmc {
namespace {

// ─── UTF-8 helpers ───

inline char32_t utf8_to_char32(const std::string& s, size_t& pos)
{
    unsigned char c = static_cast<unsigned char>(s[pos]);
    if (c < 0x80) { ++pos; return static_cast<char32_t>(c); }
    if ((c & 0xE0) == 0xC0 && pos + 1 < s.size()) {
        char32_t cp = (static_cast<char32_t>(c & 0x1F) << 6) |
                      static_cast<char32_t>(static_cast<unsigned char>(s[pos + 1]) & 0x3F);
        pos += 2; return cp;
    }
    if ((c & 0xF0) == 0xE0 && pos + 2 < s.size()) {
        char32_t cp = (static_cast<char32_t>(c & 0x0F) << 12) |
                      (static_cast<char32_t>(static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 6) |
                      static_cast<char32_t>(static_cast<unsigned char>(s[pos + 2]) & 0x3F);
        pos += 3; return cp;
    }
    if ((c & 0xF8) == 0xF0 && pos + 3 < s.size()) {
        char32_t cp = (static_cast<char32_t>(c & 0x07) << 18) |
                      (static_cast<char32_t>(static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 12) |
                      (static_cast<char32_t>(static_cast<unsigned char>(s[pos + 2]) & 0x3F) << 6) |
                      static_cast<char32_t>(static_cast<unsigned char>(s[pos + 3]) & 0x3F);
        pos += 4; return cp;
    }
    ++pos; return 0xFFFD;
}

inline std::string char32_to_utf8(char32_t cp)
{
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

// ─── Unicode character classification (range-table lookup) ───

struct UnicodeRange { char32_t lo, hi; };

inline bool in_ranges(char32_t cp, const UnicodeRange* ranges, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (cp >= ranges[i].lo && cp <= ranges[i].hi) return true;
    }
    return false;
}

inline bool is_control_char(char32_t cp)
{
    if (cp == '\t' || cp == '\n' || cp == '\r') return false;
    return (cp <= 0x1F) || (cp >= 0x7F && cp <= 0x9F);
}

constexpr UnicodeRange kWhitespaceRanges[] = {
    {' ', ' '}, {'\t', '\r'},  // space, tab, LF, VT, FF, CR
    {0x00A0, 0x00A0}, {0x1680, 0x1680},
    {0x2000, 0x200A}, {0x2028, 0x2029},
    {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000},
};

inline bool is_whitespace(char32_t cp)
{
    return in_ranges(cp, kWhitespaceRanges,
        sizeof(kWhitespaceRanges) / sizeof(kWhitespaceRanges[0]));
}

constexpr UnicodeRange kPunctuationRanges[] = {
    {33, 47}, {58, 64}, {91, 96}, {123, 126},  // ASCII punctuation
    {0x2000, 0x206F},   // General Punctuation
    {0x3000, 0x303F},   // CJK Symbols and Punctuation
    {0xFE30, 0xFE4F},   // CJK Compatibility Forms
    {0xFE50, 0xFE6F},   // Small Form Variants
    {0xFF00, 0xFF0F},   // Fullwidth
    {0xFF1A, 0xFF20}, {0xFF3B, 0xFF40}, {0xFF5B, 0xFF65},
};

inline bool is_punctuation(char32_t cp)
{
    return in_ranges(cp, kPunctuationRanges,
        sizeof(kPunctuationRanges) / sizeof(kPunctuationRanges[0]));
}

constexpr UnicodeRange kCjkRanges[] = {
    {0x4E00, 0x9FFF}, {0x3400, 0x4DBF},
    {0x20000, 0x2A6DF}, {0x2A700, 0x2B73F},
    {0x2B740, 0x2B81F}, {0x2B820, 0x2CEAF},
    {0xF900, 0xFAFF}, {0x2F800, 0x2FA1F},
};

inline bool is_cjk_char(char32_t cp)
{
    return in_ranges(cp, kCjkRanges,
        sizeof(kCjkRanges) / sizeof(kCjkRanges[0]));
}

constexpr UnicodeRange kMnRanges[] = {
    {0x0300, 0x036F},   // Combining Diacritical Marks
    {0x1AB0, 0x1AFF},   // Extended
    {0x1DC0, 0x1DFF},   // Supplement
    {0x20D0, 0x20FF},   // For Symbols
    {0xFE20, 0xFE2F},   // Combining Half Marks
};

inline bool is_mn_category(char32_t cp)
{
    return in_ranges(cp, kMnRanges,
        sizeof(kMnRanges) / sizeof(kMnRanges[0]));
}

// Simple NFD decomposition for common accented Latin characters
inline void nfd_decompose(char32_t cp, std::vector<char32_t>& out)
{
    // Pre-composed Latin letters → base + combining mark
    // This covers the most common accented characters
    struct Decomposition { char32_t composed; char32_t base; char32_t mark; };
    static const Decomposition kDecomps[] = {
        {0xC0, 'A', 0x0300}, {0xC1, 'A', 0x0301}, {0xC2, 'A', 0x0302}, {0xC3, 'A', 0x0303},
        {0xC4, 'A', 0x0308}, {0xC5, 'A', 0x030A}, {0xC7, 'C', 0x0327}, {0xC8, 'E', 0x0300},
        {0xC9, 'E', 0x0301}, {0xCA, 'E', 0x0302}, {0xCB, 'E', 0x0308}, {0xCC, 'I', 0x0300},
        {0xCD, 'I', 0x0301}, {0xCE, 'I', 0x0302}, {0xCF, 'I', 0x0308}, {0xD1, 'N', 0x0303},
        {0xD2, 'O', 0x0300}, {0xD3, 'O', 0x0301}, {0xD4, 'O', 0x0302}, {0xD5, 'O', 0x0303},
        {0xD6, 'O', 0x0308}, {0xD9, 'U', 0x0300}, {0xDA, 'U', 0x0301}, {0xDB, 'U', 0x0302},
        {0xDC, 'U', 0x0308}, {0xDD, 'Y', 0x0301},
        {0xE0, 'a', 0x0300}, {0xE1, 'a', 0x0301}, {0xE2, 'a', 0x0302}, {0xE3, 'a', 0x0303},
        {0xE4, 'a', 0x0308}, {0xE5, 'a', 0x030A}, {0xE7, 'c', 0x0327}, {0xE8, 'e', 0x0300},
        {0xE9, 'e', 0x0301}, {0xEA, 'e', 0x0302}, {0xEB, 'e', 0x0308}, {0xEC, 'i', 0x0300},
        {0xED, 'i', 0x0301}, {0xEE, 'i', 0x0302}, {0xEF, 'i', 0x0308}, {0xF1, 'n', 0x0303},
        {0xF2, 'o', 0x0300}, {0xF3, 'o', 0x0301}, {0xF4, 'o', 0x0302}, {0xF5, 'o', 0x0303},
        {0xF6, 'o', 0x0308}, {0xF9, 'u', 0x0300}, {0xFA, 'u', 0x0301}, {0xFB, 'u', 0x0302},
        {0xFC, 'u', 0x0308}, {0xFD, 'y', 0x0301}, {0xFF, 'y', 0x0308},
    };
    for (const auto& d : kDecomps) {
        if (cp == d.composed) {
            out.push_back(d.base);
            out.push_back(d.mark);
            return;
        }
    }
    out.push_back(cp);
}

// Simple ASCII lowercase (handles A-Z only; full Unicode tolower is complex)
inline char32_t to_lower(char32_t cp)
{
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    // Latin-1 Supplement uppercase
    if (cp >= 0xC0 && cp <= 0xD6) return cp + 32;
    if (cp >= 0xD8 && cp <= 0xDE) return cp + 32;
    return cp;
}

// ─── BertNormalizer ───

struct BertNormalizerConfig {
    bool clean_text = true;
    bool handle_chinese_chars = true;
    bool lowercase = false;
    bool strip_accents_flag = false;
    bool strip_accents_set = false; // whether strip_accents was explicitly set
};

std::string bert_clean_text(const std::string& text)
{
    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        char32_t cp = utf8_to_char32(text, pos);
        if (cp == 0 || cp == 0xFFFD || is_control_char(cp)) {
            if (cp == '\t' || cp == '\n' || cp == '\r') {
                result += ' ';
            }
            continue;
        }
        result += char32_to_utf8(cp);
    }
    return result;
}

std::string bert_handle_chinese(const std::string& text)
{
    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        char32_t cp = utf8_to_char32(text, pos);
        if (is_cjk_char(cp)) {
            result += ' ';
            result += char32_to_utf8(cp);
            result += ' ';
        } else {
            result += char32_to_utf8(cp);
        }
    }
    return result;
}

std::string bert_lowercase(const std::string& text)
{
    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        char32_t cp = utf8_to_char32(text, pos);
        result += char32_to_utf8(to_lower(cp));
    }
    return result;
}

std::string bert_strip_accents(const std::string& text)
{
    // NFD decompose, then remove Mn category characters
    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        char32_t cp = utf8_to_char32(text, pos);
        std::vector<char32_t> decomposed;
        nfd_decompose(cp, decomposed);
        for (char32_t dcp : decomposed) {
            if (!is_mn_category(dcp)) {
                result += char32_to_utf8(dcp);
            }
        }
    }
    return result;
}

std::string bert_normalize(const std::string& text, const BertNormalizerConfig& cfg)
{
    std::string result = text;
    if (cfg.clean_text)          result = bert_clean_text(result);
    if (cfg.handle_chinese_chars) result = bert_handle_chinese(result);
    if (cfg.lowercase)           result = bert_lowercase(result);
    // strip_accents: if explicitly set, use that; otherwise strip when lowercase is true
    bool do_strip = cfg.strip_accents_set ? cfg.strip_accents_flag : cfg.lowercase;
    if (do_strip)                result = bert_strip_accents(result);
    return result;
}

// ─── BertPreTokenizer ───
// Splits on whitespace and punctuation. Each punctuation char becomes its own token.

std::vector<std::string> bert_pre_tokenize(const std::string& text)
{
    std::vector<std::string> tokens;
    std::string current;
    size_t pos = 0;

    while (pos < text.size()) {
        size_t start = pos;
        char32_t cp = utf8_to_char32(text, pos);

        if (is_whitespace(cp)) {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
            continue;
        }

        if (is_punctuation(cp)) {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
            tokens.push_back(text.substr(start, pos - start));
            continue;
        }

        current += text.substr(start, pos - start);
    }

    if (!current.empty()) {
        tokens.push_back(std::move(current));
    }

    return tokens;
}

// ─── WordPieceTokenizer ───

class WordPieceTokenizer final : public ITokenizer {
public:
    static std::unique_ptr<WordPieceTokenizer> Create(
        const char* json_data, std::size_t json_size, bool add_special_tokens)
    {
        auto tok = std::unique_ptr<WordPieceTokenizer>(new WordPieceTokenizer());
        tok->mAddSpecialTokens = add_special_tokens;
        tok->parse_tokenizer_json(json_data, json_size);
        return tok;
    }

    std::vector<int32_t> encode(const std::string& text) const override
    {
        if (text.empty()) {
            if (!mAddSpecialTokens) return {};
            return make_special_frame({});
        }

        std::string normalized = bert_normalize(text, mNormConfig);
        auto words = bert_pre_tokenize(normalized);

        std::vector<int32_t> ids;
        for (const auto& word : words) {
            tokenize_word(word, ids);
        }

        if (mAddSpecialTokens) {
            ids = make_special_frame(ids);
        }
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids) const override
    {
        std::string result;
        for (int32_t id : ids) {
            if (mDecodeSkipIds.count(id)) continue;
            std::string token = token_for_id(id);
            if (token.empty()) continue;

            if (token.size() >= mContinuingPrefix.size()
                && token.compare(0, mContinuingPrefix.size(), mContinuingPrefix) == 0) {
                result += token.substr(mContinuingPrefix.size());
            } else {
                if (!result.empty()) result += ' ';
                result += token;
            }
        }
        return result;
    }

    int32_t id_for_token(std::string_view token) const override
    {
        auto it = mTokenToId.find(std::string(token));
        return it != mTokenToId.end() ? it->second : -1;
    }

    std::string token_for_id(int32_t id) const override
    {
        if (id >= 0 && static_cast<size_t>(id) < mIdToToken.size()) {
            return mIdToToken[id];
        }
        return "";
    }

private:
    WordPieceTokenizer() = default;

    // ─── Greedy longest-match WordPiece encoding ───

    void tokenize_word(const std::string& word, std::vector<int32_t>& ids) const
    {
        if (word.empty()) return;

        // Count UTF-8 codepoints for max_input_chars_per_word check
        size_t char_count = 0;
        {
            size_t p = 0;
            while (p < word.size()) {
                utf8_to_char32(word, p);
                ++char_count;
            }
        }
        if (static_cast<int32_t>(char_count) > mMaxCharsPerWord) {
            ids.push_back(mUnkId);
            return;
        }

        std::vector<int32_t> sub_ids;
        size_t start = 0;

        while (start < word.size()) {
            size_t end = word.size();
            bool found = false;

            while (end > start) {
                std::string substr = word.substr(start, end - start);
                if (start > 0) substr = mContinuingPrefix + substr;

                auto it = mTokenToId.find(substr);
                if (it != mTokenToId.end()) {
                    sub_ids.push_back(it->second);
                    found = true;
                    start = end;
                    break;
                }

                // Shrink by one UTF-8 codepoint from the end
                end = shrink_utf8(word, start, end);
            }

            if (!found) {
                ids.push_back(mUnkId);
                return;
            }
        }

        ids.insert(ids.end(), sub_ids.begin(), sub_ids.end());
    }

    // Shrink end position by one UTF-8 codepoint
    static size_t shrink_utf8(const std::string& s, size_t start, size_t end)
    {
        if (end <= start) return start;
        // Walk backwards to find the start of the last codepoint
        size_t pos = end - 1;
        while (pos > start && (static_cast<unsigned char>(s[pos]) & 0xC0) == 0x80) {
            --pos;
        }
        return pos;
    }

    std::vector<int32_t> make_special_frame(std::vector<int32_t> ids) const
    {
        std::vector<int32_t> result;
        if (mClsId >= 0) result.push_back(mClsId);
        result.insert(result.end(), ids.begin(), ids.end());
        if (mSepId >= 0) result.push_back(mSepId);
        return result;
    }

    // ─── JSON parsing ───

    void parse_tokenizer_json(const char* json_data, std::size_t json_size)
    {
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(json_data, json_data + json_size);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("Failed to parse tokenizer.json: ") + e.what());
        }

        validate_model_type(j);
        parse_model_config(j);
        parse_vocab(j);
        parse_normalizer(j);
        parse_added_tokens(j);
        resolve_special_ids();
        parse_post_processor(j);
    }

    static void validate_model_type(const nlohmann::json& j)
    {
        if (!j.contains("model"))
            throw std::runtime_error("Invalid tokenizer.json: missing model");

        auto& model = j["model"];

        if (model.contains("type")) {
            if (model["type"] != "WordPiece")
                throw std::runtime_error("Not a WordPiece tokenizer");
        } else {
            // Some HF tokenizer.json files omit model.type for WordPiece.
            // Detect WordPiece by: has vocab, no merges, has continuing_subword_prefix
            bool has_vocab = model.contains("vocab");
            bool has_merges = model.contains("merges");
            bool has_prefix = model.contains("continuing_subword_prefix");
            if (!has_vocab || has_merges || !has_prefix)
                throw std::runtime_error("Not a WordPiece tokenizer (no model.type and missing WordPiece indicators)");
        }

        if (!model.contains("vocab"))
            throw std::runtime_error("Invalid tokenizer.json: missing model.vocab");
    }

    void parse_model_config(const nlohmann::json& j)
    {
        auto& model = j["model"];
        mUnkToken = model.value("unk_token", "[UNK]");
        mContinuingPrefix = model.value("continuing_subword_prefix", "##");
        mMaxCharsPerWord = model.value("max_input_chars_per_word", 100);
    }

    void parse_vocab(const nlohmann::json& j)
    {
        auto& vocab_obj = j["model"]["vocab"];
        size_t vocab_size = vocab_obj.size();
        mIdToToken.resize(vocab_size);

        for (auto& [token, id] : vocab_obj.items()) {
            int32_t token_id = id.get<int32_t>();
            if (token_id >= 0 && token_id < static_cast<int32_t>(vocab_size)) {
                mIdToToken[token_id] = token;
                mTokenToId[token] = token_id;
            }
        }
    }

    void parse_normalizer(const nlohmann::json& j)
    {
        if (!j.contains("normalizer") || j["normalizer"].is_null()) return;
        auto& norm = j["normalizer"];
        std::string norm_type = norm.value("type", "");
        if (norm_type != "BertNormalizer" && norm_type != "Sequence") return;

        if (norm_type == "Sequence") {
            // Some models wrap normalizer in a Sequence
            parse_sequence_normalizer(norm);
            return;
        }

        mNormConfig.clean_text = norm.value("clean_text", true);
        mNormConfig.handle_chinese_chars = norm.value("handle_chinese_chars", true);
        mNormConfig.lowercase = norm.value("lowercase", false);
        if (norm.contains("strip_accents") && !norm["strip_accents"].is_null()) {
            mNormConfig.strip_accents_flag = norm["strip_accents"].get<bool>();
            mNormConfig.strip_accents_set = true;
        }
    }

    void parse_sequence_normalizer(const nlohmann::json& norm)
    {
        if (!norm.contains("normalizers")) return;
        for (auto& sub : norm["normalizers"]) {
            std::string sub_type = sub.value("type", "");
            if (sub_type == "BertNormalizer") {
                mNormConfig.clean_text = sub.value("clean_text", true);
                mNormConfig.handle_chinese_chars = sub.value("handle_chinese_chars", true);
                mNormConfig.lowercase = sub.value("lowercase", false);
                if (sub.contains("strip_accents") && !sub["strip_accents"].is_null()) {
                    mNormConfig.strip_accents_flag = sub["strip_accents"].get<bool>();
                    mNormConfig.strip_accents_set = true;
                }
            }
        }
    }

    void parse_added_tokens(const nlohmann::json& j)
    {
        if (!j.contains("added_tokens")) return;
        for (auto& tok : j["added_tokens"]) {
            std::string content = tok["content"].get<std::string>();
            int32_t id = tok["id"].get<int32_t>();

            if (id >= 0 && static_cast<size_t>(id) >= mIdToToken.size()) {
                mIdToToken.resize(static_cast<size_t>(id) + 1);
            }
            if (id >= 0) {
                mIdToToken[id] = content;
                mTokenToId[content] = id;
            }

            if (tok.value("special", false)) {
                mSpecialIds.insert(id);
            }
        }
    }

    void resolve_special_ids()
    {
        auto find_id = [this](const std::string& token) -> int32_t {
            auto it = mTokenToId.find(token);
            return it != mTokenToId.end() ? it->second : -1;
        };

        mUnkId = find_id(mUnkToken);

        // All special tokens (for general use)
        if (mUnkId >= 0) mSpecialIds.insert(mUnkId);
    }

    // Extract ID from a post_processor cls/sep array: ["<token>", id]
    static int32_t extract_pp_id(const nlohmann::json& pp, const char* key)
    {
        if (pp.contains(key) && pp[key].is_array() && pp[key].size() >= 2)
            return pp[key][1].get<int32_t>();
        return -1;
    }

    // Try to find CLS/SEP from post_processor (BERT, RoBERTa, Template styles)
    void parse_cls_sep_from_post_processor(const nlohmann::json& j)
    {
        if (!j.contains("post_processor") || j["post_processor"].is_null()) return;
        auto& pp = j["post_processor"];
        std::string pp_type = pp.value("type", "");
        // All known styles use the same cls/sep array format
        if (pp_type == "TemplateProcessing" || pp_type == "BertProcessing"
            || pp_type == "RobertaProcessing") {
            mClsId = extract_pp_id(pp, "cls");
            mSepId = extract_pp_id(pp, "sep");
        }
    }

    // Resolve remaining special token IDs and build skip sets
    void resolve_pad_mask_and_skip_sets()
    {
        auto find_id = [this](const std::string& a, const std::string& b) -> int32_t {
            auto it = mTokenToId.find(a);
            if (it != mTokenToId.end()) return it->second;
            it = mTokenToId.find(b);
            return it != mTokenToId.end() ? it->second : -1;
        };

        if (mClsId < 0) mClsId = find_id("[CLS]", "<s>");
        if (mSepId < 0) mSepId = find_id("[SEP]", "</s>");
        mPadId = find_id("[PAD]", "<pad>");
        mMaskId = find_id("[MASK]", "<mask>");

        for (int32_t id : {mClsId, mSepId, mPadId}) {
            if (id >= 0) { mSpecialIds.insert(id); mDecodeSkipIds.insert(id); }
        }
        if (mMaskId >= 0) mSpecialIds.insert(mMaskId);
    }

    void parse_post_processor(const nlohmann::json& j)
    {
        parse_cls_sep_from_post_processor(j);
        resolve_pad_mask_and_skip_sets();
    }

    // ─── Data members ───

    std::vector<std::string> mIdToToken;
    std::unordered_map<std::string, int32_t> mTokenToId;
    std::unordered_set<int32_t> mSpecialIds;
    std::unordered_set<int32_t> mDecodeSkipIds; // tokens to filter during decode

    std::string mUnkToken = "[UNK]";
    std::string mContinuingPrefix = "##";
    int32_t mMaxCharsPerWord = 100;
    bool mAddSpecialTokens = true;

    BertNormalizerConfig mNormConfig;

    int32_t mClsId = -1;
    int32_t mSepId = -1;
    int32_t mPadId = -1;
    int32_t mMaskId = -1;
    int32_t mUnkId = -1;
};

} // namespace

std::unique_ptr<ITokenizer> CreateWordPieceTokenizer(
    const char* tokenizer_json_data,
    std::size_t tokenizer_json_size,
    bool add_special_tokens)
{
    return WordPieceTokenizer::Create(
        tokenizer_json_data, tokenizer_json_size, add_special_tokens);
}

} // namespace trtmc
