/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/kolibri1/runtime/tokenizer.h"

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

// ─── BpeTokenizer implementation ───

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
        if (mIsSentencePiece) {
            encode_sentencepiece(normalized, result);
        } else if (mIsMetaspace) {
            encode_metaspace(normalized, result);
        } else {
            encode_bytelevel(normalized, result);
        }
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
        switch (mDecoderType) {
        case DecoderType::kByteLevel:
            return byte_decode(joined);
        case DecoderType::kMetaspace:
            return decode_metaspace(joined);
        case DecoderType::kSequence:
            return decode_sequence(joined);
        }
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

    enum class DecoderType { kByteLevel, kMetaspace, kSequence };

    std::string normalize_text(const std::string& source) const {
        std::string text = source;
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
            if (mNormalizeWhitespace && u_isUWhiteSpace(static_cast<UChar32>(cp))) {
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
    void encode_metaspace(const std::string& text, std::vector<int32_t>& result) const {
        std::vector<std::string> chars;
        const char* cp = text.data();
        const char* ce = cp + text.size();
        while (cp < ce) {
            const char* cs = cp;
            read_utf8(cp, ce);
            std::string ch(cs, cp);
            if (mTokenToId.count(ch)) {
                chars.push_back(std::move(ch));
            }
        }
        auto tokens = apply_merges(std::move(chars));
        for (const auto& token : tokens) {
            auto it = mTokenToId.find(token);
            if (it != mTokenToId.end()) {
                result.push_back(it->second);
            }
        }
    }

    // SentencePiece-style encode: replace spaces with ▁, split to chars, BPE merge.
    // Used for Metaspace pre-tokenizer and Sequence decoder models (LLaMA, Mistral, Phi-3).
    // Normalize text for SentencePiece: replace spaces with ▁, handle prepend
    std::string normalize_sentencepiece(const std::string& text) const {
        static const std::string sp = "\xe2\x96\x81"; // U+2581
        std::string out;
        if (mSentencePiecePrependAlways) {
            out = sp;
        }
        for (char c : text) {
            out += (c == ' ') ? sp : std::string(1, c);
        }
        // Metaspace prepend_scheme=first: prepend if not already starting with ▁
        if (!mSentencePiecePrependAlways && mSentencePiecePrependIfMissing &&
            (out.empty() || out.compare(0, sp.size(), sp) != 0)) {
            out = sp + out;
        }
        return out;
    }

    void encode_sentencepiece(const std::string& text, std::vector<int32_t>& result) const {
        std::string normalized = normalize_sentencepiece(text);

        // Split into UTF-8 characters
        std::vector<std::string> chars;
        const char* cp = normalized.data();
        const char* ce = cp + normalized.size();
        while (cp < ce) {
            const char* cs = cp;
            read_utf8(cp, ce);
            chars.emplace_back(cs, cp);
        }

        // byte_fallback: replace unknown chars with <0xXX>
        if (mByteFallback) {
            chars = apply_byte_fallback_encode(std::move(chars));
        }

        // BPE merge and lookup
        auto tokens = apply_merges(std::move(chars));
        for (const auto& token : tokens) {
            auto it = mTokenToId.find(token);
            if (it != mTokenToId.end()) {
                result.push_back(it->second);
            }
        }
    }

    // For byte_fallback: replace chars not in vocab with <0xXX> byte tokens
    std::vector<std::string> apply_byte_fallback_encode(std::vector<std::string> chars) const {
        std::vector<std::string> result;
        for (auto& ch : chars) {
            if (mTokenToId.count(ch)) {
                result.push_back(std::move(ch));
            } else {
                // Split into individual bytes as <0xXX>
                for (unsigned char byte : ch) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "<0x%02X>", byte);
                    result.push_back(std::string(buf));
                }
            }
        }
        return result;
    }

    void encode_bytelevel(const std::string& text, std::vector<int32_t>& result) const {
        std::vector<std::string> words;
        if (mRegex) {
            UErrorCode status = U_ZERO_ERROR;
            const std::string prepared =
                mAddPrefixSpace && !text.empty() && text.front() != ' ' ? " " + text : text;
            const auto unicode = icu::UnicodeString::fromUTF8(prepared);
            std::unique_ptr<icu::RegexMatcher> matcher(mRegex->matcher(unicode, status));
            if (U_FAILURE(status))
                throw std::runtime_error("Kolibri1 tokenizer regex matcher failed");
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
                throw std::runtime_error("Kolibri1 tokenizer regex matching failed");
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

    // ─── Sequence decoder (SentencePiece BPE models: LLaMA, Mistral, Phi-3) ───

    std::string decode_sequence(const std::string& joined) const {
        std::string text = joined;
        // Step 1: Apply Replace operations (e.g. ▁ → space)
        for (const auto& rep : mSeqDecoderReplaces) {
            text = string_replace_all(text, rep.pattern, rep.content);
        }
        // Step 2: ByteFallback — convert <0xXX> tokens to raw bytes
        if (mSeqDecoderByteFallback) {
            text = apply_byte_fallback(text);
        }
        // Step 3: Fuse is implicit (tokens already joined)
        // Step 4: Strip leading space
        if (mSeqDecoderStripLeft && !text.empty() && text[0] == ' ') {
            text.erase(0, 1);
        }
        return text;
    }

    static std::string string_replace_all(const std::string& input, const std::string& from,
                                          const std::string& to) {
        if (from.empty())
            return input;
        std::string result;
        result.reserve(input.size());
        size_t pos = 0;
        while (pos < input.size()) {
            if (pos + from.size() <= input.size() && input.compare(pos, from.size(), from) == 0) {
                result += to;
                pos += from.size();
            } else {
                result.push_back(input[pos]);
                ++pos;
            }
        }
        return result;
    }

    static int hex_char_value(char c) {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        return -1;
    }

    // Try to parse <0xXX> at position pos. Returns parsed byte or -1.
    static int try_parse_byte_token(const std::string& text, size_t pos) {
        if (pos + 6 > text.size())
            return -1;
        if (text[pos] != '<' || text[pos + 1] != '0' || text[pos + 2] != 'x' ||
            text[pos + 5] != '>')
            return -1;
        int h = hex_char_value(text[pos + 3]);
        int l = hex_char_value(text[pos + 4]);
        if (h < 0 || l < 0)
            return -1;
        return (h << 4) | l;
    }

    static std::string apply_byte_fallback(const std::string& text) {
        std::string result;
        result.reserve(text.size());
        size_t pos = 0;
        while (pos < text.size()) {
            int byte_val = try_parse_byte_token(text, pos);
            if (byte_val >= 0) {
                result.push_back(static_cast<char>(byte_val));
                pos += 6;
            } else {
                result.push_back(text[pos]);
                ++pos;
            }
        }
        return result;
    }

    // ─── Byte-level encoding ───

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
        // Repair only after concatenating every token's bytes: UTF-8 can span tokens.
        std::string repaired;
        icu::UnicodeString::fromUTF8(result).toUTF8String(repaired);
        return repaired;
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

    DecoderType classify_decoder_type(const nlohmann::json& j) const {
        if (!j.contains("decoder") || j["decoder"].is_null()) {
            return mIsMetaspace ? DecoderType::kMetaspace : DecoderType::kByteLevel;
        }
        std::string dt = j["decoder"].value("type", "");
        if (dt == "ByteLevel")
            return DecoderType::kByteLevel;
        if (dt == "Metaspace")
            return DecoderType::kMetaspace;
        if (dt == "Sequence")
            return DecoderType::kSequence;
        return mIsMetaspace ? DecoderType::kMetaspace : DecoderType::kByteLevel;
    }

    void detect_decoder(const nlohmann::json& j) {
        mDecoderType = classify_decoder_type(j);
        if (mDecoderType == DecoderType::kSequence) {
            parse_sequence_decoder(j["decoder"]);
        }
        // SentencePiece encode: vocab uses ▁ (U+2581) for spaces.
        // Detect by: (1) ▁ in vocabulary, or (2) normalizer prepends ▁.
        static const std::string spiece_marker = "\xe2\x96\x81"; // U+2581
        mIsSentencePiece = mTokenToId.count(spiece_marker) > 0 || mSentencePiecePrependAlways;
    }

    void parse_seq_decoder_replace(const nlohmann::json& sub) {
        SeqDecoderReplace rep;
        if (sub.contains("pattern") && sub["pattern"].contains("String")) {
            rep.pattern = sub["pattern"]["String"].get<std::string>();
        }
        rep.content = sub.value("content", "");
        if (!rep.pattern.empty()) {
            mSeqDecoderReplaces.push_back(std::move(rep));
        }
    }

    void parse_sequence_decoder(const nlohmann::json& dec) {
        if (!dec.contains("decoders"))
            return;
        for (auto& sub : dec["decoders"]) {
            std::string sub_type = sub.value("type", "");
            if (sub_type == "Replace") {
                parse_seq_decoder_replace(sub);
            } else if (sub_type == "ByteFallback") {
                mSeqDecoderByteFallback = true;
            } else if (sub_type == "Strip") {
                if (sub.value("content", " ") == " " && sub.value("start", 0) > 0) {
                    mSeqDecoderStripLeft = true;
                }
            }
            // Fuse is implicit (tokens already joined)
        }
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

        mByteFallback = model.value("byte_fallback", false);
        mEndOfWordSuffix = optional_model_string(model, "end_of_word_suffix");
        parse_vocab(j);
        parse_merges(j);
        parse_added_tokens(j);
        parse_post_processor(j);
        detect_normalizer(j);
        const auto& pretokenizer = j.at("pre_tokenizer");
        if (pretokenizer.value("type", "") != "Sequence" ||
            pretokenizer.at("pretokenizers").size() != 2)
            throw std::runtime_error("Expected Kolibri1 Split and ByteLevel pre-tokenizer");
        const auto& split = pretokenizer.at("pretokenizers")[0];
        const auto& bytes = pretokenizer.at("pretokenizers")[1];
        if (split.value("type", "") != "Split" || split.value("behavior", "") != "Isolated" ||
            split.value("invert", true) || bytes.value("type", "") != "ByteLevel" ||
            bytes.value("use_regex", true) || bytes.value("add_prefix_space", true))
            throw std::runtime_error("Unsupported Kolibri1 pre-tokenizer options");
        mAddPrefixSpace = false;
        const auto pattern = split.at("pattern").at("Regex").get<std::string>();
        UErrorCode regex_status = U_ZERO_ERROR;
        mRegex.reset(
            icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(pattern), 0, regex_status));
        if (U_FAILURE(regex_status))
            throw std::runtime_error("Kolibri1 tokenizer regex compilation failed");
        detect_decoder(j);
    }

    // Detect normalizer: check for Prepend (always prepend ▁) vs none
    static bool normalizer_sequence_prepends(const nlohmann::json& norm,
                                             const std::string& norm_type) {
        if (norm_type != "Sequence" || !norm.contains("normalizers"))
            return false;
        for (auto& sub : norm["normalizers"]) {
            if (sub.value("type", "") == "Prepend")
                return true;
        }
        return false;
    }

    static bool normalizer_replaces_space_with_sentence_piece(const nlohmann::json& norm,
                                                              const std::string& norm_type) {
        if (norm_type == "Sequence" && norm.contains("normalizers")) {
            for (const auto& sub : norm["normalizers"]) {
                if (normalizer_replaces_space_with_sentence_piece(sub, sub.value("type", ""))) {
                    return true;
                }
            }
            return false;
        }
        if (norm_type != "Replace")
            return false;
        if (!norm.contains("pattern") || !norm["pattern"].contains("String"))
            return false;
        if (norm["pattern"]["String"].get<std::string>() != " ")
            return false;
        return norm.value("content", "") == "\xe2\x96\x81";
    }

    void detect_normalizer(const nlohmann::json& j) {
        if (!j.contains("normalizer") || j["normalizer"].is_null())
            return;
        auto& norm = j["normalizer"];
        std::string norm_type = norm.value("type", "");
        detect_text_normalizers(norm);
        if (normalizer_sequence_prepends(norm, norm_type)) {
            mSentencePiecePrependAlways = true;
        } else if (normalizer_replaces_space_with_sentence_piece(norm, norm_type)) {
            // Gemma replaces spaces with ▁ but does not prepend ▁ to the
            // first token. Preserve the old prepend-if-missing behavior for
            // Metaspace-style SentencePiece models.
            mSentencePiecePrependIfMissing = false;
        }
    }

    void detect_text_normalizers(const nlohmann::json& normalizer) {
        const std::string type = normalizer.value("type", "");
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
                throw std::runtime_error("Kolibri1 tokenizer replacement cannot be empty");
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
    bool mIsMetaspace = false; // Set by detect_pre_tokenizer.
    bool mIsSentencePiece = false;
    bool mSentencePiecePrependAlways =
        false; // true for Normalizer Prepend, false for Metaspace first
    bool mSentencePiecePrependIfMissing = true;
    bool mByteFallback = false;
    bool mAddPrefixSpace{false};
    std::unique_ptr<icu::RegexPattern> mRegex;
    std::vector<std::pair<std::string, std::string>> mLiteralReplacements;
    bool mNormalizeLowercase = false;
    bool mNormalizeWhitespace = false;
    std::string mEndOfWordSuffix;

    // Post-processor: BOS/EOS token IDs to add when add_special_tokens=true
    // Vectors to support multiple BOS/EOS tokens (e.g. GLM-4: [gMASK] + <sop>)
    std::vector<int32_t> mPostBosIds;
    std::vector<int32_t> mPostEosIds;

    DecoderType mDecoderType = DecoderType::kByteLevel;

    // Sequence decoder config (parsed from tokenizer.json decoder field)
    struct SeqDecoderReplace {
        std::string pattern;
        std::string content;
    };
    std::vector<SeqDecoderReplace> mSeqDecoderReplaces;
    bool mSeqDecoderByteFallback = false;
    bool mSeqDecoderStripLeft = false;
};

} // namespace

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const char* tokenizer_json_data,
                                               std::size_t tokenizer_json_size,
                                               bool add_special_tokens) {
    return BpeTokenizer::Create(tokenizer_json_data, tokenizer_json_size, add_special_tokens);
}

} // namespace trtmc
