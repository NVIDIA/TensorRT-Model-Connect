/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gte/runtime/tokenizer.h"
#include "families/gte/runtime/unicode_data.h"

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

// ─── Unicode character classification (range-table lookup) ───

struct UnicodeRange {
    char32_t lo, hi;
};

inline bool in_ranges(char32_t cp, const UnicodeRange* ranges, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (cp >= ranges[i].lo && cp <= ranges[i].hi)
            return true;
    }
    return false;
}

constexpr UnicodeRange kOtherControlRanges[] = {
    {0x00AD, 0x00AD},   {0x0600, 0x0605},   {0x061C, 0x061C},   {0x06DD, 0x06DD},
    {0x070F, 0x070F},   {0x180E, 0x180E},   {0x200B, 0x200F},   {0x202A, 0x202E},
    {0x2060, 0x2064},   {0x2066, 0x206F},   {0xE000, 0xF8FF},   {0xFEFF, 0xFEFF},
    {0xFFF9, 0xFFFB},   {0x110BD, 0x110BD}, {0x1BCA0, 0x1BCA3}, {0x1D173, 0x1D17A},
    {0xE0001, 0xE0001}, {0xE0020, 0xE007F}, {0xF0000, 0xFFFFD}, {0x100000, 0x10FFFD},
};

inline bool is_control_char(char32_t cp) {
    if (cp == '\t' || cp == '\n' || cp == '\r')
        return false;
    return (cp <= 0x1F) || (cp >= 0x7F && cp <= 0x9F) ||
           in_ranges(cp, kOtherControlRanges,
                     sizeof(kOtherControlRanges) / sizeof(kOtherControlRanges[0]));
}

constexpr UnicodeRange kWhitespaceRanges[] = {
    {' ', ' '},       {'\t', '\r'}, // space, tab, LF, VT, FF, CR
    {0x00A0, 0x00A0}, {0x1680, 0x1680}, {0x2000, 0x200A}, {0x2028, 0x2029},
    {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000},
};

inline bool is_whitespace(char32_t cp) {
    return in_ranges(cp, kWhitespaceRanges,
                     sizeof(kWhitespaceRanges) / sizeof(kWhitespaceRanges[0]));
}

// Note (Jiaxin Deng): Whole blocks split non-punctuation letters and symbols.
constexpr UnicodeRange kPunctuationRanges[] = {
    {0x0021, 0x002F},   {0x003A, 0x0040},   {0x005B, 0x0060},   {0x007B, 0x007E},
    {0x00A1, 0x00A1},   {0x00A7, 0x00A7},   {0x00AB, 0x00AB},   {0x00B6, 0x00B7},
    {0x00BB, 0x00BB},   {0x00BF, 0x00BF},   {0x037E, 0x037E},   {0x0387, 0x0387},
    {0x055A, 0x055F},   {0x0589, 0x058A},   {0x05BE, 0x05BE},   {0x05C0, 0x05C0},
    {0x05C3, 0x05C3},   {0x05C6, 0x05C6},   {0x05F3, 0x05F4},   {0x0609, 0x060A},
    {0x060C, 0x060D},   {0x061B, 0x061B},   {0x061E, 0x061F},   {0x066A, 0x066D},
    {0x06D4, 0x06D4},   {0x0700, 0x070D},   {0x07F7, 0x07F9},   {0x0830, 0x083E},
    {0x085E, 0x085E},   {0x0964, 0x0965},   {0x0970, 0x0970},   {0x0AF0, 0x0AF0},
    {0x0DF4, 0x0DF4},   {0x0E4F, 0x0E4F},   {0x0E5A, 0x0E5B},   {0x0F04, 0x0F12},
    {0x0F14, 0x0F14},   {0x0F3A, 0x0F3D},   {0x0F85, 0x0F85},   {0x0FD0, 0x0FD4},
    {0x0FD9, 0x0FDA},   {0x104A, 0x104F},   {0x10FB, 0x10FB},   {0x1360, 0x1368},
    {0x1400, 0x1400},   {0x166D, 0x166E},   {0x169B, 0x169C},   {0x16EB, 0x16ED},
    {0x1735, 0x1736},   {0x17D4, 0x17D6},   {0x17D8, 0x17DA},   {0x1800, 0x180A},
    {0x1944, 0x1945},   {0x1A1E, 0x1A1F},   {0x1AA0, 0x1AA6},   {0x1AA8, 0x1AAD},
    {0x1B5A, 0x1B60},   {0x1BFC, 0x1BFF},   {0x1C3B, 0x1C3F},   {0x1C7E, 0x1C7F},
    {0x1CC0, 0x1CC7},   {0x1CD3, 0x1CD3},   {0x2010, 0x2027},   {0x2030, 0x2043},
    {0x2045, 0x2051},   {0x2053, 0x205E},   {0x207D, 0x207E},   {0x208D, 0x208E},
    {0x2308, 0x230B},   {0x2329, 0x232A},   {0x2768, 0x2775},   {0x27C5, 0x27C6},
    {0x27E6, 0x27EF},   {0x2983, 0x2998},   {0x29D8, 0x29DB},   {0x29FC, 0x29FD},
    {0x2CF9, 0x2CFC},   {0x2CFE, 0x2CFF},   {0x2D70, 0x2D70},   {0x2E00, 0x2E2E},
    {0x2E30, 0x2E42},   {0x3001, 0x3003},   {0x3008, 0x3011},   {0x3014, 0x301F},
    {0x3030, 0x3030},   {0x303D, 0x303D},   {0x30A0, 0x30A0},   {0x30FB, 0x30FB},
    {0xA4FE, 0xA4FF},   {0xA60D, 0xA60F},   {0xA673, 0xA673},   {0xA67E, 0xA67E},
    {0xA6F2, 0xA6F7},   {0xA874, 0xA877},   {0xA8CE, 0xA8CF},   {0xA8F8, 0xA8FA},
    {0xA8FC, 0xA8FC},   {0xA92E, 0xA92F},   {0xA95F, 0xA95F},   {0xA9C1, 0xA9CD},
    {0xA9DE, 0xA9DF},   {0xAA5C, 0xAA5F},   {0xAADE, 0xAADF},   {0xAAF0, 0xAAF1},
    {0xABEB, 0xABEB},   {0xFD3E, 0xFD3F},   {0xFE10, 0xFE19},   {0xFE30, 0xFE52},
    {0xFE54, 0xFE61},   {0xFE63, 0xFE63},   {0xFE68, 0xFE68},   {0xFE6A, 0xFE6B},
    {0xFF01, 0xFF03},   {0xFF05, 0xFF0A},   {0xFF0C, 0xFF0F},   {0xFF1A, 0xFF1B},
    {0xFF1F, 0xFF20},   {0xFF3B, 0xFF3D},   {0xFF3F, 0xFF3F},   {0xFF5B, 0xFF5B},
    {0xFF5D, 0xFF5D},   {0xFF5F, 0xFF65},   {0x10100, 0x10102}, {0x1039F, 0x1039F},
    {0x103D0, 0x103D0}, {0x1056F, 0x1056F}, {0x10857, 0x10857}, {0x1091F, 0x1091F},
    {0x1093F, 0x1093F}, {0x10A50, 0x10A58}, {0x10A7F, 0x10A7F}, {0x10AF0, 0x10AF6},
    {0x10B39, 0x10B3F}, {0x10B99, 0x10B9C}, {0x11047, 0x1104D}, {0x110BB, 0x110BC},
    {0x110BE, 0x110C1}, {0x11140, 0x11143}, {0x11174, 0x11175}, {0x111C5, 0x111C9},
    {0x111CD, 0x111CD}, {0x111DB, 0x111DB}, {0x111DD, 0x111DF}, {0x11238, 0x1123D},
    {0x112A9, 0x112A9}, {0x114C6, 0x114C6}, {0x115C1, 0x115D7}, {0x11641, 0x11643},
    {0x1173C, 0x1173E}, {0x12470, 0x12474}, {0x16A6E, 0x16A6F}, {0x16AF5, 0x16AF5},
    {0x16B37, 0x16B3B}, {0x16B44, 0x16B44}, {0x1BC9F, 0x1BC9F}, {0x1DA87, 0x1DA8B},
};

inline bool is_punctuation(char32_t cp) {
    if (cp < 0x80)
        return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
               (cp >= 123 && cp <= 126);
    const auto* end =
        kPunctuationRanges + sizeof(kPunctuationRanges) / sizeof(kPunctuationRanges[0]);
    const auto* found = std::lower_bound(
        kPunctuationRanges, end, cp,
        [](const UnicodeRange& range, char32_t value) { return range.hi < value; });
    return found != end && found->lo <= cp;
}

constexpr UnicodeRange kCjkRanges[] = {
    {0x4E00, 0x9FFF},   {0x3400, 0x4DBF},   {0x20000, 0x2A6DF}, {0x2A700, 0x2B73F},
    {0x2B740, 0x2B81F}, {0x2B820, 0x2CEAF}, {0xF900, 0xFAFF},   {0x2F800, 0x2FA1F},
};

inline bool is_cjk_char(char32_t cp) {
    return in_ranges(cp, kCjkRanges, sizeof(kCjkRanges) / sizeof(kCjkRanges[0]));
}

inline bool is_mn_category(char32_t cp) {
    if (cp < 0x0300)
        return false;
    const auto* begin = gte::unicode_data::kNonspacingMarks;
    const auto* end = begin + sizeof(gte::unicode_data::kNonspacingMarks) / sizeof(*begin);
    const auto* found = std::lower_bound(
        begin, end, cp, [](const auto& range, char32_t value) { return range.hi < value; });
    return found != end && found->lo <= cp;
}

inline void nfd_decompose(char32_t cp, std::vector<char32_t>& out) {
    if (cp < 0x00C0) {
        out.push_back(cp);
        return;
    }
    if (cp >= 0xAC00 && cp <= 0xD7A3) {
        const auto offset = cp - 0xAC00;
        out.push_back(0x1100 + offset / 588);
        out.push_back(0x1161 + (offset % 588) / 28);
        if (offset % 28 != 0)
            out.push_back(0x11A7 + offset % 28);
        return;
    }
    const auto* begin = gte::unicode_data::kDecompositions;
    const auto* end = begin + sizeof(gte::unicode_data::kDecompositions) / sizeof(*begin);
    const auto* found = std::lower_bound(
        begin, end, cp, [](const auto& entry, char32_t value) { return entry.composed < value; });
    if (found == end || found->composed != cp) {
        out.push_back(cp);
        return;
    }
    nfd_decompose(found->first, out);
    if (found->second != 0)
        nfd_decompose(found->second, out);
}

inline char32_t to_lower(char32_t cp) {
    if (cp < 0x80)
        return cp >= 'A' && cp <= 'Z' ? cp + 32 : cp;
    const auto* begin = gte::unicode_data::kLowercase;
    const auto* end = begin + sizeof(gte::unicode_data::kLowercase) / sizeof(*begin);
    const auto* found = std::lower_bound(
        begin, end, cp, [](const auto& entry, char32_t value) { return entry.upper < value; });
    return found != end && found->upper == cp ? found->lower : cp;
}

struct BertNormalizerConfig {
    bool clean_text = true;
    bool handle_chinese_chars = true;
    bool lowercase = false;
    bool strip_accents_flag = false;
    bool strip_accents_set = false; // whether strip_accents was explicitly set
};

std::string bert_clean_text(const std::string& text) {
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

std::string bert_handle_chinese(const std::string& text) {
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

std::string bert_lowercase(const std::string& text) {
    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        char32_t cp = utf8_to_char32(text, pos);
        if (cp == 0x0130)
            result += "i\xcc\x87";
        else
            result += char32_to_utf8(to_lower(cp));
    }
    return result;
}

std::string bert_strip_accents(const std::string& text) {
    std::vector<char32_t> decomposed;
    size_t pos = 0;
    while (pos < text.size()) {
        char32_t cp = utf8_to_char32(text, pos);
        nfd_decompose(cp, decomposed);
    }
    const auto combining_class = [](char32_t cp) {
        const auto* begin = gte::unicode_data::kCombiningClasses;
        const auto* end = begin + sizeof(gte::unicode_data::kCombiningClasses) / sizeof(*begin);
        const auto* found = std::lower_bound(
            begin, end, cp, [](const auto& entry, char32_t value) { return entry.hi < value; });
        return found != end && found->lo <= cp ? found->value : 0;
    };
    size_t segment_begin = 0;
    for (size_t i = 0; i <= decomposed.size(); ++i) {
        if (i == decomposed.size() || combining_class(decomposed[i]) == 0) {
            std::stable_sort(
                decomposed.begin() + segment_begin, decomposed.begin() + i,
                [&](char32_t a, char32_t b) { return combining_class(a) < combining_class(b); });
            segment_begin = i + 1;
        }
    }
    std::string result;
    for (char32_t cp : decomposed) {
        if (!is_mn_category(cp))
            result += char32_to_utf8(cp);
    }
    return result;
}

std::string bert_normalize(const std::string& text, const BertNormalizerConfig& cfg) {
    std::string result = text;
    if (cfg.clean_text)
        result = bert_clean_text(result);
    if (cfg.handle_chinese_chars)
        result = bert_handle_chinese(result);
    bool do_strip = cfg.strip_accents_set ? cfg.strip_accents_flag : cfg.lowercase;
    if (do_strip)
        result = bert_strip_accents(result);
    if (cfg.lowercase)
        result = bert_lowercase(result);
    return result;
}

std::vector<std::string> bert_pre_tokenize(const std::string& text) {
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
    static std::unique_ptr<WordPieceTokenizer> Create(const char* json_data, std::size_t json_size,
                                                      bool add_special_tokens) {
        auto tok = std::unique_ptr<WordPieceTokenizer>(new WordPieceTokenizer());
        tok->mAddSpecialTokens = add_special_tokens;
        tok->parse_tokenizer_json(json_data, json_size);
        return tok;
    }

    std::vector<int32_t> encode(const std::string& text) const override {
        if (text.empty()) {
            if (!mAddSpecialTokens)
                return {};
            return make_special_frame({});
        }

        std::vector<int32_t> ids;
        size_t start = 0;
        while (start < text.size()) {
            size_t next = std::string::npos;
            int32_t special_id = -1;
            size_t special_size = 0;
            for (int32_t id : mRawSpecialIds) {
                const auto& token = mIdToToken[id];
                const auto pos = text.find(token, start);
                if (pos < next || (pos == next && token.size() > special_size)) {
                    next = pos;
                    special_id = id;
                    special_size = token.size();
                }
            }
            encode_text(text.substr(start, next == std::string::npos ? next : next - start), ids);
            if (next == std::string::npos)
                break;
            ids.push_back(special_id);
            start = next + special_size;
        }

        if (mAddSpecialTokens) {
            ids = make_special_frame(ids);
        }
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string result;
        for (int32_t id : ids) {
            if (mDecodeSkipIds.count(id))
                continue;
            std::string token = token_for_id(id);
            if (token.empty())
                continue;

            if (token.size() >= mContinuingPrefix.size() &&
                token.compare(0, mContinuingPrefix.size(), mContinuingPrefix) == 0) {
                result += token.substr(mContinuingPrefix.size());
            } else {
                if (!result.empty())
                    result += ' ';
                result += token;
            }
        }
        return result;
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
    WordPieceTokenizer() = default;

    // ─── Greedy longest-match WordPiece encoding ───

    void encode_text(const std::string& text, std::vector<int32_t>& ids) const {
        auto words = bert_pre_tokenize(bert_normalize(text, mNormConfig));
        for (const auto& word : words)
            tokenize_word(word, ids);
    }

    void tokenize_word(const std::string& word, std::vector<int32_t>& ids) const {
        if (word.empty())
            return;

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
                if (start > 0)
                    substr = mContinuingPrefix + substr;

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
    static size_t shrink_utf8(const std::string& s, size_t start, size_t end) {
        if (end <= start)
            return start;
        // Walk backwards to find the start of the last codepoint
        size_t pos = end - 1;
        while (pos > start && (static_cast<unsigned char>(s[pos]) & 0xC0) == 0x80) {
            --pos;
        }
        return pos;
    }

    std::vector<int32_t> make_special_frame(std::vector<int32_t> ids) const {
        std::vector<int32_t> result;
        if (mClsId >= 0)
            result.push_back(mClsId);
        result.insert(result.end(), ids.begin(), ids.end());
        if (mSepId >= 0)
            result.push_back(mSepId);
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
        parse_model_config(j);
        parse_vocab(j);
        parse_normalizer(j);
        parse_added_tokens(j);
        resolve_special_ids();
        parse_post_processor(j);
    }

    static void validate_model(const nlohmann::json& j) {
        if (!j.contains("model"))
            throw std::runtime_error("Invalid tokenizer.json: missing model");

        auto& model = j["model"];

        if (!model.contains("vocab") || !model["vocab"].is_object())
            throw std::runtime_error("Invalid tokenizer.json: model.vocab must be an object");
    }

    void parse_model_config(const nlohmann::json& j) {
        auto& model = j["model"];
        mUnkToken = model.value("unk_token", "[UNK]");
        mContinuingPrefix = model.value("continuing_subword_prefix", "##");
        mMaxCharsPerWord = model.value("max_input_chars_per_word", 100);
    }

    void parse_vocab(const nlohmann::json& j) {
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

    void parse_normalizer(const nlohmann::json& j) {
        if (!j.contains("normalizer") || j["normalizer"].is_null())
            return;
        auto& norm = j["normalizer"];
        std::string norm_type = norm.value("type", "");
        if (norm_type != "BertNormalizer" && norm_type != "Sequence")
            return;

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

    void parse_sequence_normalizer(const nlohmann::json& norm) {
        if (!norm.contains("normalizers"))
            return;
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

    void parse_added_tokens(const nlohmann::json& j) {
        if (!j.contains("added_tokens"))
            return;
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
                // BERT's raw special tokens must bypass lowercasing and punctuation splitting.
                if (id >= 0 && !content.empty() && !tok.value("normalized", false) &&
                    !tok.value("single_word", false))
                    mRawSpecialIds.push_back(id);
            }
        }
    }

    void resolve_special_ids() {
        auto find_id = [this](const std::string& token) -> int32_t {
            auto it = mTokenToId.find(token);
            return it != mTokenToId.end() ? it->second : -1;
        };

        mUnkId = find_id(mUnkToken);

        // All special tokens (for general use)
        if (mUnkId >= 0)
            mSpecialIds.insert(mUnkId);
    }

    // Extract ID from a post_processor cls/sep array: ["<token>", id]
    static int32_t extract_pp_id(const nlohmann::json& pp, const char* key) {
        if (pp.contains(key) && pp[key].is_array() && pp[key].size() >= 2)
            return pp[key][1].get<int32_t>();
        return -1;
    }

    // Try to find CLS/SEP from post_processor (BERT, RoBERTa, Template styles)
    void parse_cls_sep_from_post_processor(const nlohmann::json& j) {
        if (!j.contains("post_processor") || j["post_processor"].is_null())
            return;
        auto& pp = j["post_processor"];
        std::string pp_type = pp.value("type", "");
        // All known styles use the same cls/sep array format
        if (pp_type == "TemplateProcessing" || pp_type == "BertProcessing" ||
            pp_type == "RobertaProcessing") {
            mClsId = extract_pp_id(pp, "cls");
            mSepId = extract_pp_id(pp, "sep");
        }
    }

    // Resolve remaining special token IDs and build skip sets
    void resolve_pad_mask_and_skip_sets() {
        auto find_id = [this](const std::string& a, const std::string& b) -> int32_t {
            auto it = mTokenToId.find(a);
            if (it != mTokenToId.end())
                return it->second;
            it = mTokenToId.find(b);
            return it != mTokenToId.end() ? it->second : -1;
        };

        if (mClsId < 0)
            mClsId = find_id("[CLS]", "<s>");
        if (mSepId < 0)
            mSepId = find_id("[SEP]", "</s>");
        mPadId = find_id("[PAD]", "<pad>");
        mMaskId = find_id("[MASK]", "<mask>");

        for (int32_t id : {mClsId, mSepId, mPadId}) {
            if (id >= 0) {
                mSpecialIds.insert(id);
                mDecodeSkipIds.insert(id);
            }
        }
        if (mMaskId >= 0)
            mSpecialIds.insert(mMaskId);
    }

    void parse_post_processor(const nlohmann::json& j) {
        parse_cls_sep_from_post_processor(j);
        resolve_pad_mask_and_skip_sets();
    }

    // ─── Data members ───

    std::vector<std::string> mIdToToken;
    std::unordered_map<std::string, int32_t> mTokenToId;
    std::unordered_set<int32_t> mSpecialIds;
    std::vector<int32_t> mRawSpecialIds;
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

std::unique_ptr<ITokenizer> CreateWordPieceTokenizer(const char* tokenizer_json_data,
                                                     std::size_t tokenizer_json_size,
                                                     bool add_special_tokens) {
    return WordPieceTokenizer::Create(tokenizer_json_data, tokenizer_json_size, add_special_tokens);
}

} // namespace trtmc
