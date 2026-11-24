/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vibevoice/runtime/tokenizer.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <list>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <string>
#include <unicode/normalizer2.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc::vibevoice {
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
    // Check the bytes left before forming p + n: a pointer past the end is undefined.
    const auto left = end - p;
    if ((c & 0xE0) == 0xC0 && left >= 2) {
        char32_t cp =
            (static_cast<char32_t>(c & 0x1F) << 6) | (static_cast<unsigned char>(p[1]) & 0x3F);
        p += 2;
        return cp;
    }
    if ((c & 0xF0) == 0xE0 && left >= 3) {
        char32_t cp = (static_cast<char32_t>(c & 0x0F) << 12) |
                      (static_cast<char32_t>(static_cast<unsigned char>(p[1]) & 0x3F) << 6) |
                      (static_cast<unsigned char>(p[2]) & 0x3F);
        p += 3;
        return cp;
    }
    if ((c & 0xF8) == 0xF0 && left >= 4) {
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

enum class Variant { kLlama, kQwen3, kBloom, kDeepSeek, kClip };

// ── Character classification ──

// \p{L} and \p{N} as the declared Split pattern means them: ICU's general category,
// so every script's letters and every decimal digit, letter number and other number
// (Arabic-Indic and Devanagari digits, superscripts, fractions) are classified.
inline bool is_letter(char32_t cp) {
    return (U_GET_GC_MASK(static_cast<UChar32>(cp)) & U_GC_L_MASK) != 0;
}

inline bool is_digit(char32_t cp) {
    return (U_GET_GC_MASK(static_cast<UChar32>(cp)) & U_GC_N_MASK) != 0;
}

inline bool is_whitespace(char32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C // VT, FF
           || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) // en quad through hair space
           || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
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
// Greedy whitespace ends at the last newline, including intervening spaces/tabs.
// Leave any whitespace after that newline for the next regex branch.
inline void scan_qwen3_newline_chunk(char32_t first_cp, const char*& p, const char* end) {
    const char* last_newline = is_newline(first_cp) ? p : nullptr;
    const char* scan = p;
    while (scan < end) {
        const auto cp = read_utf8(scan, end);
        if (!is_whitespace(cp))
            break;
        if (is_newline(cp))
            last_newline = scan;
    }
    if (last_newline)
        p = last_newline;
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

inline bool is_two_char_contraction(char32_t c1, char32_t c2) {
    return (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l');
}

inline char32_t contraction_case(char32_t cp, bool ignore_case) {
    if (!ignore_case)
        return cp;
    if (cp >= 'A' && cp <= 'Z')
        return cp + ('a' - 'A');
    // Long s is the only non-ASCII case-fold equivalent in these suffixes.
    return cp == 0x17F ? 's' : cp;
}

// Returns the UTF-8 byte length of a contraction suffix after the apostrophe.
inline int match_contraction_suffix(const char* p, const char* end, bool ignore_case) {
    if (p >= end)
        return 0;
    const char* next = p;
    const auto c = contraction_case(read_utf8(next, end), ignore_case);
    if (c == 's' || c == 't' || c == 'm' || c == 'd')
        return static_cast<int>(next - p);
    if (next < end &&
        is_two_char_contraction(c, contraction_case(read_utf8(next, end), ignore_case)))
        return static_cast<int>(next - p);
    return 0;
}

// ── GPT-2/Qwen3 pre-tokenize dispatch helpers ──

// Qwen3 uses case-insensitive contractions, then the optional letter prefix.
inline bool try_contraction(char32_t cp, const char*& p, const char* end, const char* start,
                            Variant variant, std::vector<std::string>& result) {
    if (cp != '\'')
        return false;
    int suffix = match_contraction_suffix(p, end, variant == Variant::kQwen3);
    if (suffix > 0) {
        p += suffix;
    } else if (variant == Variant::kQwen3) {
        return false;
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
        // Only a literal space may prefix the punctuation branch of the Qwen regex.
        if (variant == Variant::kQwen3 && cp != ' ') {
            p = after_prefix;
            return false;
        }
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

// Hands each pre-token to `visit` as soon as it is cut, so a caller that stops (by
// throwing) never holds the pieces of the rest of the text.
template <class Visit>
void pre_tokenize_each(const std::string& text, Variant variant, int digit_group, Visit&& visit) {
    const char* p = text.data();
    const char* end = p + text.size();
    std::vector<std::string> step; // the pieces one step cuts
    while (p < end) {
        step.clear();
        const char* start = p;
        char32_t cp = read_utf8(p, end);

        if (!try_contraction(cp, p, end, start, variant, step) &&
            !try_prefix_run(cp, p, end, start, variant, step) &&
            !try_whitespace_run(cp, p, end, start, variant, step) &&
            !try_simple_run(cp, p, end, start, variant, step, digit_group)) {
            // Other chars (punctuation/symbols)
            scan_others(p, end);
            if (variant == pretok::Variant::kQwen3) {
                // Qwen3's ` ?[^\s\p{L}\p{N}]+[\r\n]*` keeps trailing newlines
                // attached to punctuation/symbol runs even without an optional prefix.
                scan_newlines(p, end);
            }
            step.emplace_back(start, p);
        }
        for (const auto& piece : step)
            visit(piece);
    }
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
                                                bool add_special_tokens, std::int32_t vocab_limit) {
        if (vocab_limit <= 0)
            throw std::invalid_argument("the tokenizer needs the decoder vocabulary size");
        auto tokenizer = std::unique_ptr<BpeTokenizer>(new BpeTokenizer());
        tokenizer->mAddSpecialTokens = add_special_tokens;
        tokenizer->mVocabLimit = vocab_limit;
        try {
            tokenizer->parse_tokenizer_json(tokenizer_json_data, tokenizer_json_size);
        } catch (const nlohmann::json::exception& error) {
            // A field of the wrong JSON type is a defect of the bundle, like the others.
            throw std::runtime_error(std::string("tokenizer.json is malformed: ") + error.what());
        }
        return tokenizer;
    }

    // Stops as soon as the ids exceed `max_tokens`: the merge work of text that cannot
    // fit ends after one pre-token past the budget.
    static void check_budget(const std::vector<int32_t>& result, std::size_t max_tokens) {
        if (result.size() > max_tokens)
            throw std::length_error("tokenizer input needs more than " +
                                    std::to_string(max_tokens) + " tokens");
    }

    void encode_segment(const std::string& text, std::vector<int32_t>& result,
                        std::size_t max_tokens) const {
        const auto normalized = normalize_text(text);
        if (mIsSentencePiece) {
            encode_sentencepiece(normalized, result);
        } else if (mIsMetaspace) {
            encode_metaspace(normalized, result);
        } else {
            encode_bytelevel(normalized, result, max_tokens);
        }
        check_budget(result, max_tokens);
    }

    std::vector<int32_t> encode(const std::string& text) const override {
        return encode_within(text, std::numeric_limits<std::size_t>::max());
    }

    std::size_t max_text_bytes(std::size_t max_tokens) const override {
        // NFC composes at most three UTF-8 bytes into one (U+0390 from its six-byte
        // decomposition, a Hangul syllable from three jamo).
        const std::size_t per_token = 3U * mMaxTokenBytes;
        if (max_tokens >= std::numeric_limits<std::size_t>::max() / per_token - 1U)
            return std::numeric_limits<std::size_t>::max();
        return (max_tokens + 1U) * per_token - 1U;
    }

    std::vector<int32_t> encode_within(const std::string& text,
                                       std::size_t max_tokens) const override {
        std::vector<int32_t> result;
        if (text.empty())
            return result;
        // Refused from its size alone, before it is copied, normalized or split.
        if (text.size() > max_text_bytes(max_tokens))
            throw std::length_error("tokenizer input of " + std::to_string(text.size()) +
                                    " bytes cannot fit " + std::to_string(max_tokens) + " tokens");
        // Text in, as the reference takes a Python str: invalid UTF-8 is rejected rather
        // than tokenized byte by byte or replaced by NFC, and ICU takes int32 lengths.
        if (text.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
            utf8_lossy(text) != text)
            throw std::invalid_argument("tokenizer input must be valid UTF-8 below 2 GiB");

        if (mAddSpecialTokens) {
            for (int32_t bos_id : mPostBosIds)
                result.push_back(bos_id);
        }

        auto segments = split_added_tokens(text);
        for (const auto& seg : segments) {
            if (seg.added_id >= 0)
                result.push_back(seg.added_id);
            else
                encode_segment(seg.text, result, max_tokens);
            check_budget(result, max_tokens);
        }

        if (mAddSpecialTokens) {
            for (int32_t eos_id : mPostEosIds)
                result.push_back(eos_id);
        }
        check_budget(result, max_tokens);

        return result;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        switch (mDecoderType) {
        case DecoderType::kByteLevel:
            return utf8_lossy(byte_level_bytes(ids));
        case DecoderType::kMetaspace:
            return decode_metaspace(join_vocab_tokens(ids));
        case DecoderType::kSequence:
            return decode_sequence(join_vocab_tokens(ids));
        }
        return utf8_lossy(byte_level_bytes(ids));
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

    // Unicode NFC, the normalizer the Qwen2.5 tokenizer.json declares.
    static std::string nfc(const std::string& text) {
        UErrorCode status = U_ZERO_ERROR;
        const auto* normalizer = icu::Normalizer2::getNFCInstance(status);
        if (U_FAILURE(status))
            throw std::runtime_error("cannot initialize NFC normalization");
        const auto source =
            icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), text.size()));
        if (normalizer->isNormalized(source, status) && U_SUCCESS(status))
            return text;
        status = U_ZERO_ERROR;
        icu::UnicodeString normalized;
        normalizer->normalize(source, normalized, status);
        if (U_FAILURE(status))
            throw std::runtime_error("cannot NFC-normalize tokenizer input");
        std::string result;
        normalized.toUTF8String(result);
        return result;
    }

    std::string normalize_text(const std::string& input) const {
        const std::string text = mNormalizeNfc ? nfc(input) : input;
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

    void encode_bytelevel(const std::string& text, std::vector<int32_t>& result,
                          std::size_t max_tokens) const {
        const auto encode_word = [&](const std::string& word) {
            // No token covers more than mMaxTokenBytes bytes, so a piece needs at least this
            // many: one that cannot fit the budget is refused before it is expanded and merged.
            const std::size_t fewest = (word.size() + mMaxTokenBytes - 1) / mMaxTokenBytes;
            if (fewest > max_tokens - std::min(max_tokens, result.size()))
                throw std::length_error("tokenizer input needs at least " +
                                        std::to_string(result.size() + fewest) +
                                        " tokens, more than " + std::to_string(max_tokens));
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
            check_budget(result, max_tokens);
        };
        if (!mUsePreTokenizer) {
            for (const auto& word : fallback_pre_tokenize(text))
                encode_word(word);
        } else if (mPreTokenizerVariant == pretok::Variant::kBloom) {
            for (const auto& word : pretok::bloom_pre_tokenize(text))
                encode_word(word);
        } else {
            // Pieces are cut one at a time, so text that cannot fit stops at the budget
            // without holding the pieces of the rest.
            pretok::pre_tokenize_each(text, mPreTokenizerVariant, mPreTokenizerDigitGroup,
                                      encode_word);
        }
    }

    // ─── Decoding helpers ───

    // The ByteLevel decoder maps each token on its own: a token with a character outside
    // the byte-level alphabet (an added token such as "é猫") keeps its own UTF-8 bytes.
    // UTF-8 is repaired once, over all the bytes, so a character split across byte
    // tokens still decodes.
    std::string byte_level_bytes(const std::vector<int32_t>& ids) const {
        const auto& tables = byte_tables();
        std::string bytes;
        for (int32_t id : ids) {
            if (mSpecialIds.count(id) || id < 0 || static_cast<size_t>(id) >= mVocab.size())
                continue;
            const std::string& token = mVocab[id];
            std::string mapped;
            bool every_character_mapped = true;
            for (size_t pos = 0; pos < token.size() && every_character_mapped;) {
                const auto it = tables.cp_to_byte.find(utf8_to_char32(token, pos));
                if (it == tables.cp_to_byte.end())
                    every_character_mapped = false;
                else
                    mapped.push_back(static_cast<char>(it->second));
            }
            bytes += every_character_mapped ? mapped : token;
        }
        return bytes;
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

    // The ByteLevel decoder's String::from_utf8_lossy over all token bytes: well-formed
    // sequences (so a character split across tokens) stay, and each maximal subpart
    // of an ill-formed sequence, such as the first bytes of a character cut off at
    // the end of the ids, becomes one U+FFFD.
    static std::string utf8_lossy(const std::string& bytes) {
        static const char kReplacement[] = "\xEF\xBF\xBD";
        const auto* data = reinterpret_cast<const unsigned char*>(bytes.data());
        const std::size_t size = bytes.size();
        std::string result;
        result.reserve(size);
        std::size_t i = 0;
        while (i < size) {
            const unsigned char lead = data[i];
            if (lead < 0x80) {
                result.push_back(static_cast<char>(lead));
                ++i;
                continue;
            }
            // Well-formed UTF-8 (Unicode table 3-7): continuation count and the
            // allowed range of the first continuation byte.
            std::size_t continuations = 0;
            unsigned char low = 0x80;
            unsigned char high = 0xBF;
            if (lead >= 0xC2 && lead <= 0xDF) {
                continuations = 1;
            } else if (lead == 0xE0) {
                continuations = 2;
                low = 0xA0;
            } else if ((lead >= 0xE1 && lead <= 0xEC) || lead == 0xEE || lead == 0xEF) {
                continuations = 2;
            } else if (lead == 0xED) {
                continuations = 2;
                high = 0x9F;
            } else if (lead == 0xF0) {
                continuations = 3;
                low = 0x90;
            } else if (lead >= 0xF1 && lead <= 0xF3) {
                continuations = 3;
            } else if (lead == 0xF4) {
                continuations = 3;
                high = 0x8F;
            } else {
                result += kReplacement;
                ++i;
                continue;
            }
            std::size_t next = i + 1;
            bool complete = true;
            for (std::size_t k = 0; k < continuations; ++k, ++next) {
                const unsigned char lower = k == 0 ? low : 0x80;
                const unsigned char upper = k == 0 ? high : 0xBF;
                if (next >= size || data[next] < lower || data[next] > upper) {
                    complete = false;
                    break;
                }
            }
            if (complete) {
                result.append(bytes, i, continuations + 1);
            } else {
                result += kReplacement; // the maximal subpart [i, next)
            }
            i = next;
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

    // The reference's BPE (tokenizers' Word::merge_all): always merge the lowest-ranked
    // pair, the leftmost of equal ranks, then queue the pairs the merged token forms with
    // its neighbours; an entry whose pair has changed since is skipped. O(n log n) in the
    // piece's length, where a rescan after every merge is quadratic.
    std::vector<std::string> apply_merges(std::vector<std::string> tokens) const {
        const std::size_t count = tokens.size();
        if (count <= 1)
            return tokens;
        struct Candidate {
            int rank;
            std::size_t left;
            bool operator>(const Candidate& other) const {
                return rank != other.rank ? rank > other.rank : left > other.left;
            }
        };
        // Neighbour links over the pieces; `count` marks none.
        std::vector<std::size_t> next(count);
        std::vector<std::size_t> prev(count);
        std::vector<bool> alive(count, true);
        for (std::size_t index = 0; index < count; ++index) {
            next[index] = index + 1;
            prev[index] = index == 0 ? count : index - 1;
        }
        const auto rank_of = [&](std::size_t left, std::size_t right) {
            const auto it = mMergeRank.find(std::make_pair(tokens[left], tokens[right]));
            return it == mMergeRank.end() ? -1 : it->second;
        };
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;
        for (std::size_t index = 0; index + 1 < count; ++index) {
            const int rank = rank_of(index, index + 1);
            if (rank >= 0)
                queue.push({rank, index});
        }
        while (!queue.empty()) {
            const auto top = queue.top();
            queue.pop();
            const std::size_t left = top.left;
            if (!alive[left] || next[left] >= count || rank_of(left, next[left]) != top.rank)
                continue;
            const std::size_t right = next[left];
            tokens[left] += tokens[right];
            alive[right] = false;
            next[left] = next[right];
            if (next[left] < count)
                prev[next[left]] = left;
            if (prev[left] < count) {
                const int rank = rank_of(prev[left], left);
                if (rank >= 0)
                    queue.push({rank, prev[left]});
            }
            if (next[left] < count) {
                const int rank = rank_of(left, next[left]);
                if (rank >= 0)
                    queue.push({rank, left});
            }
        }
        // The first piece only ever absorbs its right neighbours.
        std::vector<std::string> result;
        for (std::size_t index = 0; index < count; index = next[index])
            result.push_back(std::move(tokens[index]));
        return result;
    }

    // ─── JSON parsing helpers ───

    static bool is_eos_content(const std::string& s) {
        return s == "<|endoftext|>" || s == "</s>" || s == "<|end_of_text|>";
    }

    void parse_vocab(const nlohmann::json& j) {
        const auto& vocab_obj = j["model"]["vocab"];
        // Every id is a row of the decoder's embedding table: an integer below the
        // decoder vocabulary, checked before the table is sized.
        std::int64_t highest = -1;
        for (const auto& [token, id] : vocab_obj.items()) {
            if (!id.is_number_integer() || id.get<std::int64_t>() < 0 ||
                id.get<std::int64_t>() >= mVocabLimit)
                throw std::runtime_error("tokenizer.json token " + token + " has id " + id.dump() +
                                         ", outside the decoder vocabulary of " +
                                         std::to_string(mVocabLimit));
            highest = std::max(highest, id.get<std::int64_t>());
        }
        mVocab.resize(static_cast<size_t>(highest + 1));
        // One token per id: the reference merges by operand ids, so a second token on
        // an id would merge like the first one, which matching by strings cannot follow.
        std::vector<bool> taken(static_cast<size_t>(highest + 1), false);
        for (const auto& [token, id] : vocab_obj.items()) {
            const auto token_id = id.get<int32_t>();
            if (taken[static_cast<size_t>(token_id)])
                throw std::runtime_error("tokenizer.json vocabulary id " +
                                         std::to_string(token_id) +
                                         " is used by more than one token");
            taken[static_cast<size_t>(token_id)] = true;
            mVocab[static_cast<size_t>(token_id)] = token;
            mTokenToId[token] = token_id;
            // A byte-level token covers one input byte per character.
            std::size_t characters = 0;
            for (const unsigned char byte : token)
                characters += (byte & 0xC0U) != 0x80U;
            mMaxTokenBytes = std::max(mMaxTokenBytes, characters);
        }
    }

    // The merge table as the reference reads it: one entry format for the whole table
    // (pairs, or "left right" strings with exactly one space, after any "#version"
    // header lines), and each operand and the result in the vocabulary.
    void parse_merges(const nlohmann::json& j) {
        const auto& merges_arr = j["model"]["merges"];
        const auto reject = [](const nlohmann::json& entry, const char* why) {
            throw std::runtime_error("tokenizer.json merge " + entry.dump() +
                                     " is not supported: " + why);
        };
        const bool strings = !merges_arr.empty() && merges_arr[0].is_string();
        mMergeRank.reserve(merges_arr.size());
        int rank = 0;
        for (const auto& entry : merges_arr) {
            std::string first, second;
            if (strings) {
                if (!entry.is_string())
                    reject(entry, "the table mixes two entry formats");
                const auto text = entry.get<std::string>();
                if (text.rfind("#version", 0) == 0)
                    continue;
                const auto space = text.find(' ');
                if (space == std::string::npos || text.find(' ', space + 1) != std::string::npos)
                    reject(entry, "not two parts around one space");
                first = text.substr(0, space);
                second = text.substr(space + 1);
            } else {
                if (!entry.is_array() || entry.size() != 2 || !entry[0].is_string() ||
                    !entry[1].is_string())
                    reject(entry, "not a pair of strings");
                first = entry[0].get<std::string>();
                second = entry[1].get<std::string>();
            }
            if (mTokenToId.count(first) == 0 || mTokenToId.count(second) == 0 ||
                mTokenToId.count(first + second) == 0)
                reject(entry, "an operand or the result is outside the vocabulary");
            mMergeRank[std::make_pair(std::move(first), std::move(second))] = rank++;
        }
    }

    void parse_added_tokens(const nlohmann::json& j) {
        if (!j.contains("added_tokens"))
            return;
        const auto& tokens = j.at("added_tokens");
        if (!tokens.is_array())
            throw std::runtime_error("tokenizer.json added_tokens must be a list");
        // The ids the reference assigns: it re-adds the tokens by content, in order, keeping
        // an earlier added token's id, else the vocabulary id, else the next new id, which
        // counts up from the number of vocabulary entries. A listed id that differs rejects
        // the bundle, as the build does, before either mapping changes.
        std::int64_t next_id = static_cast<std::int64_t>(mTokenToId.size());
        std::unordered_map<std::string, std::int64_t> assigned;
        // The rows the vocabulary gives its tokens. The reference gives a new token the next
        // id even where a sparse vocabulary gives that row to another token, which then
        // decodes as the new token; such a bundle is rejected, as the build does.
        std::vector<bool> vocab_rows(mVocab.size(), false);
        for (const auto& entry : mTokenToId)
            vocab_rows[static_cast<std::size_t>(entry.second)] = true;
        for (const auto& tok : tokens) {
            // Checked access: a token without a string content or an integer id
            // rejects the bundle instead of reading a missing field.
            const auto content_field = tok.is_object() ? tok.find("content") : tok.end();
            const auto id_field = tok.is_object() ? tok.find("id") : tok.end();
            if (content_field == tok.end() || !content_field->is_string() ||
                id_field == tok.end() || !id_field->is_number_integer())
                throw std::runtime_error(
                    "tokenizer.json added token needs a string content and an integer id");
            // Added tokens are matched on the raw text, as written: stripping, word
            // boundaries and matching on normalized text are not implemented.
            for (const char* option : {"lstrip", "rstrip", "single_word", "normalized"}) {
                if (flag(tok, option, false))
                    unsupported("added token option", tok);
            }
            const std::string content = content_field->get<std::string>();
            if (content.empty())
                throw std::runtime_error("tokenizer.json added token needs a nonempty content");
            const std::int64_t raw_id = id_field->get<std::int64_t>();
            // Bounded before the vocabulary grows: one stray id must not size it.
            if (raw_id < 0 || raw_id >= mVocabLimit)
                throw std::runtime_error(
                    "tokenizer.json added token " + content + " has id " + std::to_string(raw_id) +
                    ", outside the decoder vocabulary of " + std::to_string(mVocabLimit));
            std::int64_t expected = next_id;
            if (const auto earlier = assigned.find(content); earlier != assigned.end())
                expected = earlier->second;
            else if (const auto row = mTokenToId.find(content); row != mTokenToId.end())
                expected = row->second;
            else
                ++next_id;
            if (raw_id != expected)
                throw std::runtime_error(
                    "tokenizer.json added token " + content + " has id " + std::to_string(raw_id) +
                    ", but the reference assigns it " + std::to_string(expected));
            const auto row = static_cast<std::size_t>(expected);
            if (row < vocab_rows.size() && vocab_rows[row] && mVocab[row] != content)
                throw std::runtime_error("tokenizer.json added token " + content + " has id " +
                                         std::to_string(expected) + ", the vocabulary row of " +
                                         mVocab[row]);
            assigned[content] = expected;
            mMaxTokenBytes = std::max(mMaxTokenBytes, content.size());
            const auto id = static_cast<int32_t>(raw_id);

            if (static_cast<size_t>(id) >= mVocab.size()) {
                mVocab.resize(static_cast<size_t>(id) + 1);
            }
            mVocab[id] = content;
            mTokenToId[content] = id;

            if (flag(tok, "special", false)) {
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
            return pretok::Variant::kLlama;
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
        return pretok::Variant::kLlama;
    }

    // Detect variant from the Split inside a Sequence pre_tokenizer.
    static pretok::Variant detect_split_variant(const nlohmann::json& pt, int& digit_group_out) {
        digit_group_out = 0;
        if (!pt.contains("pretokenizers"))
            return pretok::Variant::kLlama;
        for (auto& sub : pt["pretokenizers"]) {
            if (sub.value("type", "") != "Split")
                continue;
            if (!sub.contains("pattern") || !sub["pattern"].contains("Regex"))
                continue;
            return classify_split(sub, digit_group_out);
        }
        return pretok::Variant::kLlama;
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
        mPreTokenizerVariant = pretok::Variant::kLlama;

        if (!j.contains("pre_tokenizer") || j["pre_tokenizer"].is_null())
            return;
        auto& pt = j["pre_tokenizer"];
        std::string pt_type = pt.value("type", "");

        if (pt_type == "ByteLevel") {
            mPreTokenizerVariant = pretok::Variant::kLlama;
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
        // check_supported_layout admits only the byte-level Qwen2.5 layout, so ordinary
        // text never takes the SentencePiece path, whatever tokens the vocabulary holds.
        mIsSentencePiece = false;
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
        // nlohmann keeps the last of two equal keys, so a repeated vocabulary entry or
        // setting would silently replace the first; the bundle is refused instead, as the
        // build refuses it.
        std::vector<std::unordered_set<std::string>> open;
        const nlohmann::json::parser_callback_t unique_keys =
            [&open](int, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
                if (event == nlohmann::json::parse_event_t::object_start) {
                    open.emplace_back();
                } else if (event == nlohmann::json::parse_event_t::object_end) {
                    open.pop_back();
                } else if (event == nlohmann::json::parse_event_t::key &&
                           !open.back().insert(parsed.get<std::string>()).second) {
                    throw std::runtime_error("tokenizer.json repeats the key " +
                                             parsed.get<std::string>());
                }
                return true;
            };
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(json_data, json_data + json_size, unique_keys);
        } catch (const nlohmann::json::exception& e) {
            throw std::runtime_error(std::string("Failed to parse tokenizer.json: ") + e.what());
        }

        if (!j.contains("model"))
            throw std::runtime_error("Invalid tokenizer.json: missing model");
        auto& model = j["model"];
        if (!model.contains("vocab") || !model["vocab"].is_object())
            throw std::runtime_error("Invalid tokenizer.json: model.vocab must be an object");
        if (!model.contains("merges") || !model["merges"].is_array())
            throw std::runtime_error("Invalid tokenizer.json: model.merges must be an array");

        check_supported_layout(j);
        mByteFallback = flag(model, "byte_fallback", false);
        mEndOfWordSuffix = optional_model_string(model, "end_of_word_suffix");
        parse_vocab(j);
        parse_merges(j);
        parse_added_tokens(j);
        parse_post_processor(j);
        detect_normalizer(j);
        detect_pre_tokenizer(j);
        detect_decoder(j);
    }

    // The Split pattern of Qwen2.5's tokenizer.json, which pretok::Variant::kQwen3 implements.
    static constexpr const char* kQwenSplitPattern =
        R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";

    [[noreturn]] static void unsupported(const char* part, const nlohmann::json& value) {
        throw std::runtime_error(std::string("tokenizer.json ") + part +
                                 " is not supported: " + value.dump());
    }

    // A boolean field: absent or null counts as `fallback`.
    static bool flag(const nlohmann::json& object, const char* key, bool fallback) {
        const auto it = object.find(key);
        if (it == object.end() || it->is_null())
            return fallback;
        if (!it->is_boolean())
            unsupported(key, *it);
        return it->get<bool>();
    }

    static bool empty_string(const nlohmann::json& object, const char* key) {
        const auto it = object.find(key);
        return it == object.end() || it->is_null() ||
               (it->is_string() && it->get<std::string>().empty());
    }

    static bool is_nfc(const nlohmann::json& norm) {
        if (!norm.is_object())
            return false;
        const std::string type = norm.value("type", "");
        if (type == "NFC")
            return true;
        if (type != "Sequence" || !norm.contains("normalizers") || !norm["normalizers"].is_array())
            return false;
        for (const auto& child : norm["normalizers"]) {
            if (!is_nfc(child))
                return false;
        }
        return true;
    }

    // The layout this tokenizer reproduces exactly: Qwen2.5's, which the
    // checkpoint's processor names. NFC; the Qwen Split pattern, then ByteLevel without
    // its own regex or prefix space; a ByteLevel decoder; no truncation, padding or
    // post-processor that adds tokens; BPE without byte fallback, ignore_merges,
    // dropout, an unknown token or subword affixes. Any other layout would be tokenized
    // differently from the reference without notice, so it rejects the bundle.
    static void check_supported_layout(const nlohmann::json& j) {
        if (j.contains("normalizer") && !j["normalizer"].is_null() && !is_nfc(j["normalizer"]))
            unsupported("normalizer", j["normalizer"]);
        for (const char* field : {"truncation", "padding"}) {
            if (j.contains(field) && !j[field].is_null())
                unsupported(field, j[field]);
        }
        const auto post = j.value("post_processor", nlohmann::json());
        if (!post.is_null() && !(post.is_object() && post.value("type", "") == "ByteLevel"))
            unsupported("post_processor", post);
        const auto pre = j.value("pre_tokenizer", nlohmann::json());
        const auto steps = pre.is_object() && pre.value("type", "") == "Sequence"
                               ? pre.value("pretokenizers", nlohmann::json())
                               : nlohmann::json();
        const bool qwen =
            steps.is_array() && steps.size() == 2 && steps[0].is_object() && steps[1].is_object() &&
            steps[0].value("type", "") == "Split" && steps[0].contains("pattern") &&
            steps[0]["pattern"].is_object() &&
            steps[0]["pattern"].value("Regex", "") == kQwenSplitPattern &&
            steps[0].value("behavior", "") == "Isolated" && !flag(steps[0], "invert", false) &&
            steps[1].value("type", "") == "ByteLevel" &&
            !flag(steps[1], "add_prefix_space", true) && !flag(steps[1], "use_regex", true);
        if (!qwen)
            unsupported("pre_tokenizer", pre);
        const auto decoder = j.value("decoder", nlohmann::json());
        if (!decoder.is_object() || decoder.value("type", "") != "ByteLevel")
            unsupported("decoder", decoder);
        const auto& model = j["model"];
        if (model.value("type", "BPE") != "BPE" || flag(model, "byte_fallback", false) ||
            flag(model, "ignore_merges", false) ||
            !model.value("dropout", nlohmann::json()).is_null() ||
            !model.value("unk_token", nlohmann::json()).is_null() ||
            !empty_string(model, "continuing_subword_prefix") ||
            !empty_string(model, "end_of_word_suffix")) {
            auto settings = model;
            settings.erase("vocab");
            settings.erase("merges");
            unsupported("model", settings);
        }
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
        if (type == "NFC") {
            mNormalizeNfc = true;
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
    // The most input bytes one token covers (a vocabulary token or an added token).
    std::size_t mMaxTokenBytes = 1;

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
    bool mNormalizeNfc = false;
    // Added-token ids must lie below this: the decoder's vocabulary.
    std::int64_t mVocabLimit = 0;
    std::string mEndOfWordSuffix;

    // Post-processor: BOS/EOS token IDs to add when add_special_tokens=true
    // Vectors to support multiple BOS/EOS tokens (e.g. GLM-4: [gMASK] + <sop>)
    std::vector<int32_t> mPostBosIds;
    std::vector<int32_t> mPostEosIds;
    pretok::Variant mPreTokenizerVariant = pretok::Variant::kLlama;
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
                                               bool add_special_tokens, std::int32_t vocab_limit) {
    return BpeTokenizer::Create(tokenizer_json_data, tokenizer_json_size, add_special_tokens,
                                vocab_limit);
}

} // namespace trtmc::vibevoice
