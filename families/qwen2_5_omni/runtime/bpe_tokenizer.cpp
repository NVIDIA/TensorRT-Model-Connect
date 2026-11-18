/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen2_5_omni/runtime/tokenizer.h"

#include <algorithm>
#include <climits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
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

inline bool is_whitespace(char32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C // VT, FF
           || cp == 0xA0                   // non-breaking space
           || cp == 0x2000 || cp == 0x200A // en space through hair space
           || cp == 0x3000;                // ideographic space
}

class BpeTokenizer final : public ITokenizer {
  public:
    static std::unique_ptr<BpeTokenizer> Create(const char* tokenizer_json_data,
                                                std::size_t tokenizer_json_size,
                                                bool add_special_tokens = false) {
        auto tokenizer = std::unique_ptr<BpeTokenizer>(new BpeTokenizer());
        tokenizer->mAddSpecialTokens = add_special_tokens;
        tokenizer->parse_tokenizer_json(tokenizer_json_data, tokenizer_json_size);
        return tokenizer;
    }

    void encode_segment(const std::string& text, std::vector<int32_t>& result) const {
        const auto normalized = normalize_text(text);
        encode_bytelevel(normalized, result);
    }

    std::vector<int32_t> encode(const std::string& text) const override {
        std::vector<int32_t> result;
        if (text.empty())
            return result;

        if (mAddSpecialTokens) {
            for (int32_t bos_id : mPostBosIds)
                result.push_back(bos_id);
        }

        auto segments = split_added_tokens(text);
        for (const auto& seg : segments) {
            if (seg.added_id >= 0)
                result.push_back(seg.added_id);
            else
                encode_segment(seg.text, result);
        }

        if (mAddSpecialTokens) {
            for (int32_t eos_id : mPostEosIds)
                result.push_back(eos_id);
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

    std::string normalize_text(const std::string& source) const {
        std::string text = source;
        if (mNormalizeNfc) {
            UErrorCode status = U_ZERO_ERROR;
            const auto* normalizer = icu::Normalizer2::getNFCInstance(status);
            if (U_FAILURE(status))
                throw std::runtime_error("Qwen2.5 Omni tokenizer NFC initialization failed");
            icu::UnicodeString normalized;
            normalizer->normalize(icu::UnicodeString::fromUTF8(text), normalized, status);
            if (U_FAILURE(status))
                throw std::runtime_error("Qwen2.5 Omni tokenizer NFC normalization failed");
            text.clear();
            normalized.toUTF8String(text);
        }
        for (const auto& replacement : mLiteralReplacements) {
            std::size_t offset = 0;
            while ((offset = text.find(replacement.first, offset)) != std::string::npos) {
                text.replace(offset, replacement.first.size(), replacement.second);
                offset += replacement.second.size();
            }
        }
        if (!mNormalizeLowercase && !mNormalizeWhitespace) {
            return text;
        }

        std::string normalized;
        normalized.reserve(text.size());
        const char* cursor = text.data();
        const char* end = cursor + text.size();
        bool previous_was_whitespace = false;
        while (cursor < end) {
            const char* char_start = cursor;
            const char32_t cp = read_utf8(cursor, end);
            if (mNormalizeWhitespace && is_whitespace(cp)) {
                if (!previous_was_whitespace) {
                    normalized.push_back(' ');
                }
                previous_was_whitespace = true;
                continue;
            }
            previous_was_whitespace = false;
            if (mNormalizeLowercase && cp >= 'A' && cp <= 'Z') {
                normalized.push_back(static_cast<char>(cp - 'A' + 'a'));
            } else {
                normalized.append(char_start, cursor);
            }
        }
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
            segments.push_back({text, -1});
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

    // ─── Encoding helpers ───

    // Metaspace encode (DeepSeek style): raw UTF-8 char split, BPE merge.
    // Spaces are dropped (not in vocab), Ġ (U+0120) used as separator.
    void encode_bytelevel(const std::string& text, std::vector<int32_t>& result) const {
        std::vector<std::string> words;
        {
            UErrorCode status = U_ZERO_ERROR;
            const auto unicode = icu::UnicodeString::fromUTF8(text);
            std::unique_ptr<icu::RegexMatcher> matcher(mRegex->matcher(unicode, status));
            if (U_FAILURE(status))
                throw std::runtime_error("Qwen2.5 Omni tokenizer regex matcher failed");
            int32_t previous = 0;
            while (matcher->find(status)) {
                const int32_t start = matcher->start(status), end = matcher->end(status);
                if (start > previous) {
                    std::string gap;
                    unicode.tempSubStringBetween(previous, start).toUTF8String(gap);
                    words.push_back(std::move(gap));
                }
                if (end > start) {
                    std::string match;
                    unicode.tempSubStringBetween(start, end).toUTF8String(match);
                    words.push_back(std::move(match));
                }
                previous = end;
            }
            if (U_FAILURE(status))
                throw std::runtime_error("Qwen2.5 Omni tokenizer regex matching failed");
            if (previous < unicode.length()) {
                std::string tail;
                unicode.tempSubString(previous).toUTF8String(tail);
                words.push_back(std::move(tail));
            }
        }

        for (const auto& word : words) {
            auto chars = byte_encode(word);
            if (!chars.empty() && !mEndOfWordSuffix.empty()) {
                chars.back() += mEndOfWordSuffix;
            }
            auto tokens = apply_merges(std::move(chars));
            for (const auto& token : tokens) {
                auto it = mTokenToId.find(token);
                if (it != mTokenToId.end()) {
                    result.push_back(it->second);
                }
            }
        }
    }

    // ─── Decoding helpers ───

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
        std::string valid;
        icu::UnicodeString::fromUTF8(result).toUTF8String(valid);
        return valid;
    }

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

            if (id >= 0 && static_cast<size_t>(id) >= mVocab.size()) {
                mVocab.resize(static_cast<size_t>(id) + 1);
            }
            if (id >= 0) {
                mVocab[id] = content;
                mTokenToId[content] = id;
            }

            if (tok.value("special", false)) {
                mSpecialTokens[content] = id;
                mSpecialIds.insert(id);
                if (is_eos_content(content))
                    mEosId = id;
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

        mEndOfWordSuffix = optional_model_string(model, "end_of_word_suffix");
        parse_vocab(j);
        parse_merges(j);
        parse_added_tokens(j);
        parse_post_processor(j);
        detect_normalizer(j);
        const auto& pretokenizer = j.at("pre_tokenizer");
        const auto parts = pretokenizer.value("pretokenizers", nlohmann::json::array());
        if (pretokenizer.value("type", "") != "Sequence" || !parts.is_array() ||
            parts.size() != 2 || parts[0].value("type", "") != "Split" ||
            parts[0].value("behavior", "") != "Isolated" || parts[0].value("invert", false) ||
            !parts[0].contains("pattern") || !parts[0]["pattern"].contains("Regex") ||
            parts[1].value("type", "") != "ByteLevel" || parts[1].value("add_prefix_space", true) ||
            parts[1].value("use_regex", true) || j.at("decoder").value("type", "") != "ByteLevel")
            throw std::runtime_error(
                "Qwen2.5 Omni tokenizer requires regex Split followed by ByteLevel");
        const auto pattern = parts[0]["pattern"]["Regex"].get<std::string>();
        if (pattern.empty())
            throw std::runtime_error("Qwen2.5 Omni tokenizer regex cannot be empty");
        UErrorCode status = U_ZERO_ERROR;
        mRegex.reset(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(pattern), 0, status));
        if (U_FAILURE(status))
            throw std::runtime_error("Qwen2.5 Omni tokenizer regex compilation failed");
    }

    void detect_normalizer(const nlohmann::json& j) {
        if (j.contains("normalizer") && !j["normalizer"].is_null())
            detect_text_normalizers(j["normalizer"]);
    }

    void detect_text_normalizers(const nlohmann::json& normalizer) {
        const std::string type = normalizer.value("type", "");
        if (type == "NFC") {
            mNormalizeNfc = true;
            return;
        }
        if (type == "Sequence" && normalizer.contains("normalizers")) {
            for (const auto& child : normalizer["normalizers"]) {
                detect_text_normalizers(child);
            }
            return;
        }
        if (type == "Replace" && normalizer.contains("pattern") &&
            normalizer["pattern"].contains("String")) {
            const auto pattern = normalizer["pattern"]["String"].get<std::string>();
            if (pattern.empty())
                throw std::runtime_error("Qwen2.5 Omni tokenizer replacement cannot be empty");
            mLiteralReplacements.emplace_back(pattern, normalizer.at("content").get<std::string>());
            return;
        }
        if (type == "Lowercase") {
            mNormalizeLowercase = true;
            return;
        }
        if (type == "Replace" && normalizer.contains("pattern") &&
            normalizer["pattern"].contains("Regex") &&
            normalizer["pattern"]["Regex"].get<std::string>() == "\\s+" &&
            normalizer.value("content", "") == " ") {
            mNormalizeWhitespace = true;
        }
    }

    // Parse post_processor to extract BOS/EOS tokens for add_special_tokens
    void parse_post_processor(const nlohmann::json& j) {
        if (!j.contains("post_processor") || j["post_processor"].is_null())
            return;
        auto& pp = j["post_processor"];
        std::string pp_type = pp.value("type", "");

        if (pp_type == "TemplateProcessing") {
            parse_template_post_processor(pp);
        } else if (pp_type == "Sequence") {
            parse_sequence_post_processor(pp);
        } else if (pp_type == "RobertaProcessing") {
            parse_roberta_post_processor(pp);
        }
        // ByteLevel post_processor doesn't add tokens, nothing to do
    }

    // Iterate Sequence post_processor's processors array to find TemplateProcessing
    void parse_sequence_post_processor(const nlohmann::json& pp) {
        if (!pp.contains("processors"))
            return;
        for (auto& sub : pp["processors"]) {
            if (sub.value("type", "") == "TemplateProcessing") {
                parse_template_post_processor(sub);
                return; // first TemplateProcessing wins
            }
        }
    }

    // RobertaProcessing: {"type":"RobertaProcessing","cls":["<s>",0],"sep":["</s>",2],...}
    void parse_roberta_post_processor(const nlohmann::json& pp) {
        if (pp.contains("cls") && pp["cls"].is_array() && pp["cls"].size() >= 2) {
            int32_t cls_id = pp["cls"][1].get<int32_t>();
            mPostBosIds.push_back(cls_id);
        }
        if (pp.contains("sep") && pp["sep"].is_array() && pp["sep"].size() >= 2) {
            int32_t sep_id = pp["sep"][1].get<int32_t>();
            mPostEosIds.push_back(sep_id);
        }
    }

    int32_t resolve_special_token_id(const nlohmann::json& entry) const {
        if (!entry.contains("SpecialToken"))
            return -1;
        std::string token_id = entry["SpecialToken"].value("id", "");
        if (token_id.empty())
            return -1;
        auto it = mSpecialTokens.find(token_id);
        return it != mSpecialTokens.end() ? it->second : -1;
    }

    void parse_template_post_processor(const nlohmann::json& pp) {
        if (!pp.contains("single"))
            return;
        bool seen_sequence = false;
        for (auto& entry : pp["single"]) {
            if (entry.contains("Sequence")) {
                seen_sequence = true;
                continue;
            }
            int32_t id = resolve_special_token_id(entry);
            if (id < 0)
                continue;
            if (seen_sequence) {
                mPostEosIds.push_back(id);
            } else {
                mPostBosIds.push_back(id);
            }
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

    bool mAddSpecialTokens = false;
    std::unique_ptr<icu::RegexPattern> mRegex;
    std::vector<std::pair<std::string, std::string>> mLiteralReplacements;
    bool mNormalizeLowercase = false;
    bool mNormalizeNfc = false;
    bool mNormalizeWhitespace = false;
    std::string mEndOfWordSuffix;

    // Post-processor: BOS/EOS token IDs to add when add_special_tokens=true
    // Vectors to support multiple BOS/EOS tokens (e.g. GLM-4: [gMASK] + <sop>)
    std::vector<int32_t> mPostBosIds;
    std::vector<int32_t> mPostEosIds;
};

} // namespace

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const char* tokenizer_json_data,
                                               std::size_t tokenizer_json_size,
                                               bool add_special_tokens) {
    return BpeTokenizer::Create(tokenizer_json_data, tokenizer_json_size, add_special_tokens);
}

} // namespace trtmc
