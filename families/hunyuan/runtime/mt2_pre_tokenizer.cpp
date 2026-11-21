/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/hunyuan/runtime/mt2_pre_tokenizer.h"

#include <cstdint>
#include <stdexcept>
#include <unicode/uchar.h>
#include <unicode/utf8.h>

namespace trtmc::hunyuan {
using nlohmann::json;
namespace {
const json& expected_sequence() {
    static const json value = json::parse(R"json({
  "type": "Sequence",
  "pretokenizers": [
    {
      "type": "Split",
      "pattern": {
        "Regex": "\\p{N}{1,3}"
      },
      "behavior": "Isolated",
      "invert": false
    },
    {
      "type": "Split",
      "pattern": {
        "Regex": "[一-龥぀-ゟ゠-ヿ]+"
      },
      "behavior": "Isolated",
      "invert": false
    },
    {
      "type": "Split",
      "pattern": {
        "Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
      },
      "behavior": "Isolated",
      "invert": false
    },
    {
      "type": "ByteLevel",
      "add_prefix_space": false,
      "trim_offsets": true,
      "use_regex": false
    }
  ]
})json");
    return value;
}

bool category(uint32_t cp, char name) {
    const auto flags = U_MASK(u_charType(static_cast<UChar32>(cp)));
    switch (name) {
    case 'N':
        return flags & U_GC_N_MASK;
    case 'L':
        return flags & U_GC_L_MASK;
    case 'M':
        return flags & U_GC_M_MASK;
    case 'P':
        return flags & U_GC_P_MASK;
    case 'S':
        return flags & U_GC_S_MASK;
    case 's':
        return u_isUWhiteSpace(static_cast<UChar32>(cp));
    default:
        throw std::invalid_argument("Unsupported Hunyuan Unicode category");
    }
}

bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x309F) ||
           (cp >= 0x30A0 && cp <= 0x30FF);
}

struct Text {
    std::vector<uint32_t> codepoints;
    std::vector<std::size_t> byte_offsets{0};

    explicit Text(const std::string& original) {
        int32_t offset = 0;
        const auto length = static_cast<int32_t>(original.size());
        while (offset < length) {
            UChar32 cp;
            U8_NEXT(original.data(), offset, length, cp);
            if (cp < 0)
                throw std::invalid_argument("Invalid Hunyuan UTF-8 codepoint");
            codepoints.push_back(static_cast<uint32_t>(cp));
            byte_offsets.push_back(static_cast<std::size_t>(offset));
        }
    }
};

// These are the three exact patterns accepted by expected_sequence(), not a
// general regex engine. Linear scans avoid recursive std::regex stack overflow
// for valid inputs up to the family limit. ICU supplies Unicode categories;
// only the checkpoint's literal ASCII/CJK ranges are explicit.
enum class Stage { digits, cjk, lexical };

bool letter_or_mark(uint32_t cp) {
    return category(cp, 'L') || category(cp, 'M');
}

bool punctuation_or_symbol(uint32_t cp) {
    return category(cp, 'P') || category(cp, 'S');
}

bool ascii_letter(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

bool ascii_punctuation(uint32_t cp) {
    return (cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) || (cp >= 0x5B && cp <= 0x60) ||
           (cp >= 0x7B && cp <= 0x7E);
}

bool newline(uint32_t cp) {
    return cp == '\r' || cp == '\n';
}

// Return the end of the first nonempty alternative, or begin for no match.
std::size_t match(const std::vector<uint32_t>& cps, std::size_t begin, Stage stage) {
    const auto size = cps.size();
    auto end = begin;
    if (stage == Stage::digits) {
        while (end < size && end - begin < 3 && category(cps[end], 'N'))
            ++end;
        return end;
    }
    if (stage == Stage::cjk) {
        while (end < size && is_cjk(cps[end]))
            ++end;
        return end;
    }

    // ASCII punctuation followed by ASCII letters has first priority.
    if (ascii_punctuation(cps[begin]) && begin + 1 < size && ascii_letter(cps[begin + 1])) {
        end = begin + 2;
        while (end < size && ascii_letter(cps[end]))
            ++end;
        return end;
    }
    // Optional non-CR/LF/L/P/S prefix followed by L/M. A leading mark can
    // match without the optional prefix (the regex's greedy-prefix backtrack).
    end = begin;
    if (!letter_or_mark(cps[begin]) && !newline(cps[begin]) && !punctuation_or_symbol(cps[begin]) &&
        begin + 1 < size && letter_or_mark(cps[begin + 1]))
        ++end;
    if (letter_or_mark(cps[end])) {
        do {
            ++end;
        } while (end < size && letter_or_mark(cps[end]));
        return end;
    }
    // Optional literal ASCII space, P/S run, then any CR/LF suffix.
    end = begin + (cps[begin] == ' ' && begin + 1 < size && punctuation_or_symbol(cps[begin + 1]));
    if (punctuation_or_symbol(cps[end])) {
        do {
            ++end;
        } while (end < size && punctuation_or_symbol(cps[end]));
        while (end < size && newline(cps[end]))
            ++end;
        return end;
    }

    end = begin;
    auto last_newline = begin;
    while (end < size && category(cps[end], 's')) {
        if (newline(cps[end]))
            last_newline = end + 1;
        ++end;
    }
    // Greedy whitespace then CR/LF ends at the LAST newline, not the run end.
    if (last_newline > begin)
        return last_newline;
    // Whitespace+(?!nonspace) consumes a full end-of-piece run, otherwise
    // all but its last character. The final whitespace alternative takes one
    // when the lookahead alternative cannot match.
    if (end < size && end - begin >= 2)
        return end - 1;
    return end;
}

std::vector<std::string> split(const std::string& input, Stage stage) {
    const Text text(input);
    std::vector<std::string> pieces;
    const auto append = [&](std::size_t begin, std::size_t end) {
        if (begin < end)
            pieces.push_back(input.substr(text.byte_offsets.at(begin),
                                          text.byte_offsets.at(end) - text.byte_offsets.at(begin)));
    };
    std::size_t start = 0;
    for (std::size_t pos = 0; pos < text.codepoints.size();) {
        const auto end = match(text.codepoints, pos, stage);
        if (end == pos) {
            ++pos;
        } else {
            append(start, pos);
            append(pos, end);
            start = pos = end;
        }
    }
    append(start, text.codepoints.size());
    return pieces;
}

} // namespace

bool is_mt2_sequence(const nlohmann::json& config) {
    return config == expected_sequence();
}

void validate_mt2_tokenizer(const json& config) {
    if (config.at("normalizer") != json{{"type", "Sequence"}, {"normalizers", json::array()}} ||
        config.at("post_processor").at("type") != "ByteLevel" ||
        config.at("decoder").at("type") != "ByteLevel" || config.at("model").at("type") != "BPE")
        throw std::invalid_argument("Unsupported native Hunyuan tokenizer contract");
    const auto& model = config.at("model");
    for (const auto* field :
         {"dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix"})
        if (!model.at(field).is_null())
            throw std::invalid_argument("Unsupported native Hunyuan BPE policy");
    for (const auto* field : {"fuse_unk", "byte_fallback", "ignore_merges"})
        if (model.at(field) != false)
            throw std::invalid_argument("Unsupported native Hunyuan BPE policy");
    for (const auto& token : config.at("added_tokens"))
        for (const auto* field : {"single_word", "lstrip", "rstrip", "normalized"})
            if (token.at(field) != false)
                throw std::invalid_argument("Unsupported native Hunyuan added-token policy");
}

std::vector<std::string> split_mt2(const std::string& text) {
    constexpr std::size_t limit = 1024 * 1024;
    if (text.size() > limit)
        throw std::invalid_argument("Hunyuan tokenizer input exceeds size limit");
    std::vector<std::string> pieces{text};
    for (const auto stage : {Stage::digits, Stage::cjk, Stage::lexical}) {
        std::vector<std::string> next;
        for (const auto& piece : pieces) {
            auto output = split(piece, stage);
            next.insert(next.end(), output.begin(), output.end());
        }
        pieces = std::move(next);
    }
    return pieces;
}
} // namespace trtmc::hunyuan
