/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/eagle_vlm/runtime/tokenizer.h"

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

// ─── Hand-written BPE pre-tokenizer ───
//
// Supports two regex variants:
//
// GPT-2 (used by GPT-2, Falcon, OPT):
//   'contractions| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// Qwen3 (used by Qwen3, LLaMA-3, Mistral, Phi):
//   (?i:contractions)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//   ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Key differences: Qwen3 allows any non-CR/LF/letter/digit as optional prefix before
// letters, matches single digits, and has explicit newline handling.

namespace pretok {

enum class Variant { kGpt2, kQwen3, kBloom, kDeepSeek, kClip };

// ── Character classification ──

struct UnicodeRange {
    char32_t lo, hi;
};

constexpr UnicodeRange kLetterRanges[] = {
    {'A', 'Z'},         {'a', 'z'},       {0xB5, 0xB5},   // µ (micro sign, treated as letter)
    {0xC0, 0xD6},       {0xD8, 0xF6},     {0xF8, 0x1BA},  // Latin-1 + Extended A/B
    {0x1BC, 0x1BF},     {0x1C4, 0x293},   {0x295, 0x2AF}, // Latin Extended B cont.
    {0x370, 0x373},     {0x376, 0x377},   {0x37B, 0x37D},   {0x37F, 0x37F}, // Greek
    {0x386, 0x386},     {0x388, 0x38A},   {0x38C, 0x38C},   {0x38E, 0x3A1},
    {0x3A3, 0x3F5},     {0x3F7, 0x481},   {0x48A, 0x52F},   // Greek + Cyrillic
    {0x531, 0x556},     {0x559, 0x559},                     // Armenian
    {0x560, 0x588},                                         // Armenian lowercase
    {0x600, 0x6FF},                                         // Arabic
    {0x900, 0x97F},                                         // Devanagari
    {0xE00, 0xE7F},                                         // Thai
    {0x10A0, 0x10C5},                                       // Georgian
    {0x13A0, 0x13F5},                                       // Cherokee
    {0x1C90, 0x1CBA},   {0x1CBD, 0x1CBF},                   // Georgian Extended
    {0x1D00, 0x1D2B},   {0x1D6B, 0x1D77}, {0x1D79, 0x1D9A}, // Phonetic Extensions
    {0x1E00, 0x1F15},   {0x1F18, 0x1F1D}, {0x1F20, 0x1F45}, // Latin Ext. Additional + Greek Ext.
    {0x2C00, 0x2C5F},                                       // Glagolitic
    {0x3040, 0x309F},                                       // Hiragana
    {0x30A0, 0x30FF},                                       // Katakana
    {0x3400, 0x4DBF},                                       // CJK Extension A
    {0x4E00, 0x9FFF},                                       // CJK Unified Ideographs
    {0xAC00, 0xD7AF},                                       // Hangul Syllables
    {0xFB00, 0xFDFF},   // Alphabetic Presentation Forms + Arabic Forms
    {0x10000, 0x1007F}, // Linear B Syllabary
};

inline bool is_letter(char32_t cp) {
    for (const auto& r : kLetterRanges) {
        if (cp >= r.lo && cp <= r.hi)
            return true;
    }
    return false;
}

inline bool is_digit(char32_t cp) {
    return (cp >= '0' && cp <= '9');
}

inline bool is_whitespace(char32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C // VT, FF
           || cp == 0xA0                   // non-breaking space
           || cp == 0x2000 || cp == 0x200A // en space through hair space
           || cp == 0x3000;                // ideographic space
}

// BLOOM punctuation set: .,!?...
constexpr char32_t kBloomPunct[] = {
    '.',    ',', '!', '?',
    0x2026, // ellipsis
    0x3002, // ideographic full stop
    0xFF0C, // fullwidth comma
    0x3001, // ideographic comma
    0x0964, // Devanagari danda
    0x06D4, // Arabic full stop
    0x060C, // Arabic comma
};

inline bool is_bloom_punct(char32_t cp) {
    for (auto c : kBloomPunct) {
        if (cp == c)
            return true;
    }
    return false;
}

inline bool is_newline(char32_t cp) {
    return cp == '\n' || cp == '\r';
}

// Is this char a valid optional prefix before a word?
// GPT-2/BLOOM: only space (0x20)
// Qwen3: any char except CR, LF, letter, digit
// DeepSeek: any whitespace (\s?)
inline bool is_prefix(char32_t cp, Variant v) {
    if (v == Variant::kClip) {
        return false;
    }
    if (v == Variant::kQwen3) {
        return !is_newline(cp) && !is_letter(cp) && !is_digit(cp);
    }
    if (v == Variant::kDeepSeek) {
        return is_whitespace(cp);
    }
    return cp == ' ';
}

// BLOOM word char: not whitespace and not BLOOM punctuation
inline bool is_bloom_word_char(char32_t cp) {
    // The serialized regex is `[^(\s|[.,!?…。，、।۔،])]`. Parentheses and
    // the pipe are literal members of that negated character class.
    return !is_whitespace(cp) && !is_bloom_punct(cp) && cp != '(' && cp != ')' && cp != '|';
}

// ── Scanning helpers (advance pointer past matching chars) ──

inline void scan_letters(const char*& p, const char* end) {
    while (p < end) {
        const char* peek = p;
        if (!is_letter(read_utf8(peek, end)))
            break;
        p = peek;
    }
}

inline void scan_digits(const char*& p, const char* end, int max_group = 0) {
    int count = 0;
    while (p < end) {
        if (max_group > 0 && count >= max_group)
            break;
        const char* peek = p;
        if (!is_digit(read_utf8(peek, end)))
            break;
        p = peek;
        ++count;
    }
}

inline void scan_others(const char*& p, const char* end) {
    while (p < end) {
        const char* peek = p;
        char32_t nc = read_utf8(peek, end);
        if (is_whitespace(nc) || is_letter(nc) || is_digit(nc))
            break;
        p = peek;
    }
}

inline void scan_newlines(const char*& p, const char* end) {
    while (p < end) {
        const char* peek = p;
        if (!is_newline(read_utf8(peek, end)))
            break;
        p = peek;
    }
}

inline void scan_all_whitespace(const char*& p, const char* end) {
    while (p < end) {
        const char* peek = p;
        if (!is_whitespace(read_utf8(peek, end)))
            break;
        p = peek;
    }
}

// Qwen3 regex branch: \s*[\r\n]+
// Consume any leading whitespace up to the first newline, then consume the
// contiguous newline run itself, but stop before whitespace that follows the
// last newline.
inline void scan_qwen3_newline_chunk(char32_t first_cp, const char*& p, const char* end) {
    bool saw_newline = is_newline(first_cp);
    while (p < end) {
        const char* peek = p;
        char32_t nc = read_utf8(peek, end);
        if (!saw_newline) {
            if (is_newline(nc)) {
                saw_newline = true;
                p = peek;
                continue;
            }
            if (is_whitespace(nc)) {
                p = peek;
                continue;
            }
            break;
        }
        if (!is_newline(nc))
            break;
        p = peek;
    }
}

inline void scan_bloom_words(const char*& p, const char* end) {
    while (p < end) {
        const char* peek = p;
        if (!is_bloom_word_char(read_utf8(peek, end)))
            break;
        p = peek;
    }
}

// Emit whitespace run, leaving last ws char for next token's optional prefix
inline void emit_whitespace_leave_last(const char*& p, const char* end, const char* start,
                                       std::vector<std::string>& result) {
    const char* last_ws_start = start;
    while (p < end) {
        const char* peek = p;
        char32_t nc = read_utf8(peek, end);
        if (!is_whitespace(nc))
            break;
        last_ws_start = p;
        p = peek;
    }
    if (p < end && last_ws_start > start) {
        p = last_ws_start;
    }
    result.emplace_back(start, p);
}

// ── Contraction matching ──

inline bool is_two_char_contraction(char c1, char c2) {
    return (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l');
}

// Returns length of contraction suffix after apostrophe ('s 't 'm 'd 're 've 'll)
inline int match_contraction_suffix(const char* p, const char* end) {
    if (p >= end)
        return 0;
    char c = *p;
    if (c == 's' || c == 't' || c == 'm' || c == 'd')
        return 1;
    if (p + 1 < end && is_two_char_contraction(c, p[1]))
        return 2;
    return 0;
}

// ── GPT-2/Qwen3 pre-tokenize dispatch helpers ──

// Handle apostrophe: either contraction or "other" chars run
inline bool try_contraction(char32_t cp, const char*& p, const char* end, const char* start,
                            std::vector<std::string>& result) {
    if (cp != '\'')
        return false;
    int suffix = match_contraction_suffix(p, end);
    if (suffix > 0) {
        p += suffix;
    } else {
        scan_others(p, end);
    }
    result.emplace_back(start, p);
    return true;
}

// Handle optional prefix + letter/digit/other run
inline bool try_prefix_run(char32_t cp, const char*& p, const char* end, const char* start,
                           Variant variant, std::vector<std::string>& result) {
    if (!is_prefix(cp, variant) || p >= end)
        return false;
    const char* after_prefix = p;
    char32_t next_cp = read_utf8(p, end);

    if (is_letter(next_cp)) {
        scan_letters(p, end);
        result.emplace_back(start, p);
        return true;
    }
    if (is_digit(next_cp)) {
        // The regex prefix [^\r\n\p{L}\p{N}]? only applies before \p{L}+ (letters).
        // Digits are matched by standalone \p{N} — no prefix. Back up so the
        // digit is handled by try_simple_run instead.
        if (variant == Variant::kQwen3) {
            p = after_prefix;
            return false;
        }
        scan_digits(p, end);
        result.emplace_back(start, p);
        return true;
    }
    if (!is_whitespace(next_cp)) {
        scan_others(p, end);
        if (variant == Variant::kQwen3)
            scan_newlines(p, end);
        result.emplace_back(start, p);
        return true;
    }
    // Prefix followed by whitespace — back up, let whitespace handler deal with it
    p = after_prefix;
    return false;
}

// Check if a whitespace run contains any newline character
inline bool has_newline_in_ws(char32_t first_cp, const char* p, const char* end) {
    if (is_newline(first_cp))
        return true;
    const char* scan = p;
    while (scan < end) {
        const char* peek = scan;
        char32_t nc = read_utf8(peek, end);
        if (!is_whitespace(nc))
            break;
        if (is_newline(nc))
            return true;
        scan = peek;
    }
    return false;
}

// Handle whitespace (Qwen3 newline sequences + general whitespace)
inline bool try_whitespace_run(char32_t cp, const char*& p, const char* end, const char* start,
                               Variant variant, std::vector<std::string>& result) {
    if (!is_whitespace(cp))
        return false;

    // kClip is selected only for CLIP's Removed + inverted Split contract,
    // which retains regex matches and removes the gaps. Whitespace therefore
    // never reaches ByteLevel.
    if (variant == Variant::kClip) {
        scan_all_whitespace(p, end);
        return true;
    }

    // Qwen3: \s*[\r\n]+ — newline sequences take priority
    if (variant == Variant::kQwen3 && has_newline_in_ws(cp, p, end)) {
        scan_qwen3_newline_chunk(cp, p, end);
        result.emplace_back(start, p);
        return true;
    }

    // General whitespace: leave last ws char for next token's prefix
    emit_whitespace_leave_last(p, end, start, result);
    return true;
}

// Handle letter or digit run (no prefix)
inline bool try_simple_run(char32_t cp, const char*& p, const char* end, const char* start,
                           Variant variant, std::vector<std::string>& result, int digit_group = 0) {
    if (is_letter(cp)) {
        scan_letters(p, end);
        result.emplace_back(start, p);
        return true;
    }
    if (is_digit(cp)) {
        if (variant == Variant::kQwen3) {
            if (digit_group > 1)
                scan_digits(p, end, digit_group - 1);
        } else if (variant != Variant::kClip) {
            scan_digits(p, end);
        }
        result.emplace_back(start, p);
        return true;
    }
    return false;
}

// ── Main pre-tokenize functions ──

std::vector<std::string> pre_tokenize(const std::string& text, Variant variant,
                                      int digit_group = 0) {
    std::vector<std::string> result;
    if (text.empty())
        return result;

    const char* p = text.data();
    const char* end = p + text.size();

    while (p < end) {
        const char* start = p;
        char32_t cp = read_utf8(p, end);

        if (try_contraction(cp, p, end, start, result))
            continue;
        if (try_prefix_run(cp, p, end, start, variant, result))
            continue;
        if (try_whitespace_run(cp, p, end, start, variant, result))
            continue;
        if (try_simple_run(cp, p, end, start, variant, result, digit_group))
            continue;

        // Other chars (punctuation/symbols)
        scan_others(p, end);
        if (variant == pretok::Variant::kQwen3) {
            // Qwen3's ` ?[^\s\p{L}\p{N}]+[\r\n]*` keeps trailing newlines
            // attached to punctuation/symbol runs even without an optional prefix.
            scan_newlines(p, end);
        }
        result.emplace_back(start, p);
    }

    return result;
}

// Return the end of a BLOOM Split-regex match beginning at `start`.
// The regex is " ?[^(\s|[.,!?...])]+": an optional ASCII space followed by
// one or more non-whitespace, non-punctuation characters.
const char* bloom_match_end(const char* start, const char* end) {
    const char* p = start;
    char32_t cp = read_utf8(p, end);
    if (cp == ' ') {
        if (p >= end)
            return nullptr;
        cp = read_utf8(p, end);
        if (!is_bloom_word_char(cp))
            return nullptr;
    } else if (!is_bloom_word_char(cp)) {
        return nullptr;
    }
    scan_bloom_words(p, end);
    return p;
}

// BLOOM uses Split(..., behavior="Isolated", invert=false) followed by
// ByteLevel(use_regex=false). Regex matches are isolated, while every
// contiguous unmatched span remains a single pre-token. In particular,
// punctuation and adjacent newlines must stay together so BPE can merge
// strings such as ".\n\n".
std::vector<std::string> bloom_pre_tokenize(const std::string& text) {
    std::vector<std::string> result;
    if (text.empty())
        return result;

    const char* p = text.data();
    const char* end = p + text.size();

    while (p < end) {
        const char* start = p;
        if (const char* match_end = bloom_match_end(start, end)) {
            result.emplace_back(start, match_end);
            p = match_end;
            continue;
        }

        // Preserve one contiguous unmatched span until the next regex match.
        read_utf8(p, end);
        while (p < end && bloom_match_end(p, end) == nullptr)
            read_utf8(p, end);
        result.emplace_back(start, p);
    }

    return result;
}

} // namespace pretok

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

    std::string normalize_text(const std::string& text) const {
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
            if (mNormalizeWhitespace && pretok::is_whitespace(cp)) {
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
        if (!mUsePreTokenizer) {
            words = fallback_pre_tokenize(text);
        } else if (mPreTokenizerVariant == pretok::Variant::kBloom) {
            words = pretok::bloom_pre_tokenize(text);
        } else {
            words = pretok::pre_tokenize(text, mPreTokenizerVariant, mPreTokenizerDigitGroup);
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
        return result;
    }

    // Generic pre-tokenizer used when no GPT-2 pattern is declared.

    static std::vector<std::string> fallback_pre_tokenize(const std::string& text) {
        std::vector<std::string> result;
        if (text.empty())
            return result;
        result.push_back(text);
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
        }
        // Sort by length descending for longest-match-first
        std::sort(mAddedTokenPatterns.begin(), mAddedTokenPatterns.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }

    // Parse digit group size from regex pattern like \p{N}{1,3} → 3.
    // Returns 0 if no digit grouping found (single digit or unlimited).
    static int parse_digit_group(const std::string& regex) {
        // Search for "\p{N}{" — the standalone grouped variant, not [^\p{N}]
        auto pos = regex.find("\\p{N}{");
        if (pos == std::string::npos)
            return 0;
        auto open = pos + 6; // after "\\p{N}{"
        auto close = regex.find('}', open);
        if (close == std::string::npos)
            return 0;
        auto comma = regex.find(',', open);
        if (comma == std::string::npos || comma >= close)
            return 0;
        return std::stoi(regex.substr(comma + 1, close - comma - 1));
    }

    static bool is_clip_split_regex(const std::string& regex) {
        const bool has_boundary_token = regex.find("startoftext") != std::string::npos ||
                                        regex.find("endoftext") != std::string::npos;
        return has_boundary_token && regex.find("\\p{L}") != std::string::npos &&
               regex.find("\\p{N}") != std::string::npos;
    }

    static pretok::Variant classify_clip_split(const nlohmann::json& split) {
        const auto behavior = split.value("behavior", "");
        const bool invert = split.value("invert", false);
        if (behavior == "Removed" && invert)
            return pretok::Variant::kClip;
        if (behavior == "Isolated" && invert)
            return pretok::Variant::kGpt2;
        throw std::runtime_error("Unsupported CLIP Split contract: behavior=" + behavior +
                                 ", invert=" + (invert ? "true" : "false"));
    }

    static bool is_qwen3_split_regex(const std::string& regex) {
        return regex.find("[^\r\n") != std::string::npos ||
               regex.find("[^\\r\\n") != std::string::npos;
    }

    static bool is_bloom_split_regex(const std::string& regex) {
        return regex.find("[^(\\s") != std::string::npos ||
               regex.find("[^(\\\\s") != std::string::npos;
    }

    // Classify one Split pre-tokenizer from both its pattern and contract.
    static pretok::Variant classify_split(const nlohmann::json& split, int& digit_group_out) {
        const auto regex = split["pattern"]["Regex"].get<std::string>();
        if (is_clip_split_regex(regex))
            return classify_clip_split(split);
        if (is_qwen3_split_regex(regex)) {
            digit_group_out = parse_digit_group(regex);
            return pretok::Variant::kQwen3;
        }
        if (is_bloom_split_regex(regex)) {
            return pretok::Variant::kBloom;
        }
        if (regex.find("\\s?[A-Za-z") != std::string::npos) {
            return pretok::Variant::kDeepSeek;
        }
        return pretok::Variant::kGpt2;
    }

    // Detect variant from the Split inside a Sequence pre_tokenizer.
    static pretok::Variant detect_split_variant(const nlohmann::json& pt, int& digit_group_out) {
        digit_group_out = 0;
        if (!pt.contains("pretokenizers"))
            return pretok::Variant::kGpt2;
        for (auto& sub : pt["pretokenizers"]) {
            if (sub.value("type", "") != "Split")
                continue;
            if (!sub.contains("pattern") || !sub["pattern"].contains("Regex"))
                continue;
            return classify_split(sub, digit_group_out);
        }
        return pretok::Variant::kGpt2;
    }

    static bool is_space_split_pre_tokenizer(const nlohmann::json& pt, const std::string& pt_type) {
        if (pt_type != "Split")
            return false;
        if (!pt.contains("pattern") || !pt["pattern"].contains("String"))
            return false;
        return pt["pattern"]["String"].get<std::string>() == " ";
    }

    void detect_pre_tokenizer(const nlohmann::json& j) {
        mUsePreTokenizer = true;
        mPreTokenizerVariant = pretok::Variant::kGpt2;

        if (!j.contains("pre_tokenizer") || j["pre_tokenizer"].is_null())
            return;
        auto& pt = j["pre_tokenizer"];
        std::string pt_type = pt.value("type", "");

        if (pt_type == "ByteLevel") {
            mPreTokenizerVariant = pretok::Variant::kGpt2;
        } else if (pt_type == "Sequence") {
            int digit_group = 0;
            mPreTokenizerVariant = detect_split_variant(pt, digit_group);
            mPreTokenizerDigitGroup = digit_group;
        } else if (is_space_split_pre_tokenizer(pt, pt_type)) {
            // Gemma SentencePiece-BPE tokenizers use a direct Split(" ")
            // pre-tokenizer with spaces already normalized to U+2581.
            mUsePreTokenizer = false;
        } else if (pt_type == "Metaspace") {
            mIsMetaspace = true;
            mUsePreTokenizer = false;
        } else if (pt_type.empty()) {
            mUsePreTokenizer = false;
        } else {
            throw std::runtime_error("Unsupported pre_tokenizer type: " + pt_type);
        }
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
        detect_pre_tokenizer(j);
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

    bool mAddSpecialTokens = false;
    bool mUsePreTokenizer = true;
    bool mIsMetaspace = false; // Set by detect_pre_tokenizer.
    bool mIsSentencePiece = false;
    bool mSentencePiecePrependAlways =
        false; // true for Normalizer Prepend, false for Metaspace first
    bool mSentencePiecePrependIfMissing = true;
    bool mByteFallback = false;
    bool mNormalizeLowercase = false;
    bool mNormalizeWhitespace = false;
    std::string mEndOfWordSuffix;

    // Post-processor: BOS/EOS token IDs to add when add_special_tokens=true
    // Vectors to support multiple BOS/EOS tokens (e.g. GLM-4: [gMASK] + <sop>)
    std::vector<int32_t> mPostBosIds;
    std::vector<int32_t> mPostEosIds;
    pretok::Variant mPreTokenizerVariant = pretok::Variant::kGpt2;
    int mPreTokenizerDigitGroup = 0; // 0=unlimited, 3=\p{N}{1,3} (LLaMA 3.1/GLM-4)

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
