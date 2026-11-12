/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/glm_ocr/runtime/tokenizer.h"

#include "families/glm_ocr/runtime/unicode_tables.h"

#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::glm_ocr {
namespace {

// The checkpoint's Split pre-tokenizer pattern; pre_tokenize() implements exactly this.
constexpr std::string_view kPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

template <std::size_t N>
bool in_ranges(const unicode::Range (&ranges)[N], std::uint32_t cp) {
    const auto* end = ranges + N;
    const auto* it =
        std::upper_bound(ranges, end, cp, [](std::uint32_t value, const unicode::Range& r) {
            return value < r.first;
        });
    return it != ranges && cp <= (it - 1)->last;
}

bool is_letter(std::uint32_t cp) {
    return in_ranges(unicode::kLetters, cp);
}
bool is_number(std::uint32_t cp) {
    return in_ranges(unicode::kNumbers, cp);
}
bool is_newline(std::uint32_t cp) {
    return cp == '\r' || cp == '\n';
}

// \s of the tokenizer's regex engine: U+0009-U+000D, U+0085 and the Zs, Zl, Zp categories.
bool is_space(std::uint32_t cp) {
    return in_ranges(unicode::kSpaces, cp);
}

// Decodes one code point; returns its length or 0 for an invalid sequence.
std::size_t utf8_decode(std::string_view text, std::size_t pos, std::uint32_t& cp) {
    const auto byte = [&](std::size_t i) { return static_cast<std::uint8_t>(text[i]); };
    const std::uint8_t lead = byte(pos);
    std::size_t length = 0;
    std::uint32_t min = 0;
    if (lead < 0x80) {
        cp = lead;
        return 1;
    } else if ((lead & 0xE0) == 0xC0) {
        length = 2, cp = lead & 0x1F, min = 0x80;
    } else if ((lead & 0xF0) == 0xE0) {
        length = 3, cp = lead & 0x0F, min = 0x800;
    } else if ((lead & 0xF8) == 0xF0) {
        length = 4, cp = lead & 0x07, min = 0x10000;
    } else {
        return 0;
    }
    if (pos + length > text.size())
        return 0;
    for (std::size_t i = 1; i < length; ++i) {
        if ((byte(pos + i) & 0xC0) != 0x80)
            return 0;
        cp = (cp << 6) | (byte(pos + i) & 0x3F);
    }
    if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
        return 0;
    return length;
}

void utf8_append(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Valid sequences are copied; each maximal invalid subpart becomes one U+FFFD (the
// substitution String::from_utf8_lossy and Python's errors="replace" perform).
std::string utf8_lossy(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size());
    const auto at = [&](std::size_t i) { return static_cast<std::uint8_t>(bytes[i]); };
    std::size_t pos = 0;
    while (pos < bytes.size()) {
        const std::uint8_t lead = at(pos);
        std::size_t length = 0;
        std::uint8_t low = 0x80, high = 0xBF; // allowed range of the second byte
        if (lead < 0x80) {
            length = 1;
        } else if (lead >= 0xC2 && lead <= 0xDF) {
            length = 2;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length = 3;
            low = lead == 0xE0 ? 0xA0 : 0x80;
            high = lead == 0xED ? 0x9F : 0xBF;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length = 4;
            low = lead == 0xF0 ? 0x90 : 0x80;
            high = lead == 0xF4 ? 0x8F : 0xBF;
        }
        std::size_t valid = length == 0 ? 0 : 1;
        while (valid > 0 && valid < length && pos + valid < bytes.size()) {
            const auto byte = at(pos + valid);
            const auto lo = valid == 1 ? low : std::uint8_t{0x80};
            const auto hi = valid == 1 ? high : std::uint8_t{0xBF};
            if (byte < lo || byte > hi)
                break;
            ++valid;
        }
        if (length > 0 && valid == length) {
            out.append(bytes.substr(pos, length));
            pos += length;
        } else {
            out += "\xEF\xBF\xBD";
            pos += std::max<std::size_t>(valid, 1);
        }
    }
    return out;
}

struct CodePoints {
    std::vector<std::uint32_t> cps;
    std::vector<std::size_t> offsets; // byte offset of each code point, plus the end
};

CodePoints decode_strict(std::string_view text) {
    CodePoints result;
    result.cps.reserve(text.size());
    result.offsets.reserve(text.size() + 1);
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::uint32_t cp = 0;
        const auto length = utf8_decode(text, pos, cp);
        if (length == 0)
            throw std::invalid_argument("glm_ocr text is not valid UTF-8");
        result.cps.push_back(cp);
        result.offsets.push_back(pos);
        pos += length;
    }
    result.offsets.push_back(text.size());
    return result;
}

std::uint32_t ascii_lower(std::uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z')
        return cp + ('a' - 'A');
    return cp == 0x17F ? 's' : cp; // U+017F folds to 's' under case-insensitive matching.
}

// Length in code points of the match of kPattern at `i` (every position matches).
std::size_t match_length(const std::vector<std::uint32_t>& cps, std::size_t i) {
    const std::size_t n = cps.size();
    const std::uint32_t c = cps[i];
    const auto at = [&](std::size_t k) { return k < n ? cps[k] : 0xFFFFFFFFu; };
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c == '\'' && i + 1 < n) {
        const auto a = ascii_lower(cps[i + 1]);
        if (a == 's' || a == 't' || a == 'm' || a == 'd')
            return 2;
        if (i + 2 < n) {
            const auto b = ascii_lower(cps[i + 2]);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
                return 3;
        }
    }
    const auto letters_from = [&](std::size_t k) {
        std::size_t end = k;
        while (end < n && is_letter(cps[end]))
            ++end;
        return end;
    };
    // [^\r\n\p{L}\p{N}]?\p{L}+
    if (is_letter(c))
        return letters_from(i) - i;
    const bool prefix = !is_newline(c) && !is_number(c);
    if (prefix && i + 1 < n && is_letter(cps[i + 1]))
        return letters_from(i + 1) - i;
    // \p{N}{1,3}
    if (is_number(c)) {
        std::size_t end = i + 1;
        while (end < n && end - i < 3 && is_number(cps[end]))
            ++end;
        return end - i;
    }
    //  ?[^\s\p{L}\p{N}]+[\r\n]*
    const auto symbol = [&](std::uint32_t x) {
        return x != 0xFFFFFFFFu && !is_space(x) && !is_letter(x) && !is_number(x);
    };
    std::size_t start = (c == ' ' && symbol(at(i + 1))) ? i + 1 : i;
    if (symbol(at(start))) {
        std::size_t end = start;
        while (end < n && symbol(cps[end]))
            ++end;
        while (end < n && is_newline(cps[end]))
            ++end;
        return end - i;
    }
    // Remaining alternatives start with whitespace; `c` is whitespace here.
    std::size_t run = i;
    while (run < n && is_space(cps[run]))
        ++run;
    // \s*[\r\n]+ : up to the last line break of the whitespace run.
    for (std::size_t k = run; k > i; --k) {
        if (is_newline(cps[k - 1]))
            return k - i;
    }
    // \s+(?!\S) : the run, minus its last character when a non-space follows.
    if (run == n)
        return run - i;
    if (run - i >= 2)
        return run - 1 - i;
    // \s+
    return run - i;
}

std::string byte_level(std::string_view piece, const std::string (&table)[256]) {
    std::string mapped;
    mapped.reserve(piece.size() * 2);
    for (const char ch : piece)
        mapped += table[static_cast<std::uint8_t>(ch)];
    return mapped;
}

} // namespace

std::vector<std::string_view> pre_tokenize(std::string_view text) {
    const auto decoded = decode_strict(text);
    std::vector<std::string_view> pieces;
    std::size_t i = 0;
    while (i < decoded.cps.size()) {
        const auto length = match_length(decoded.cps, i);
        if (length == 0)
            throw std::logic_error("glm_ocr pre-tokenizer made no progress");
        const auto begin = decoded.offsets[i], end = decoded.offsets[i + length];
        pieces.push_back(text.substr(begin, end - begin));
        i += length;
    }
    return pieces;
}

Tokenizer::Tokenizer(std::string_view tokenizer_json) {
    const auto json = nlohmann::json::parse(tokenizer_json.begin(), tokenizer_json.end());
    const auto& model = json.at("model");
    if (model.at("type") != "BPE" || model.value("byte_fallback", false) ||
        !model.value("dropout", nlohmann::json()).is_null() ||
        !model.value("continuing_subword_prefix", nlohmann::json()).is_null() ||
        !model.value("end_of_word_suffix", nlohmann::json()).is_null())
        throw std::runtime_error("glm_ocr tokenizer.json is not a plain byte-level BPE model");
    if (!json.at("normalizer").is_null())
        throw std::runtime_error("glm_ocr tokenizer.json must not normalize");
    const auto& pre = json.at("pre_tokenizer");
    const auto& steps = pre.at("pretokenizers");
    if (pre.at("type") != "Sequence" || steps.size() != 2 || steps[0].at("type") != "Split" ||
        steps[0].at("pattern").at("Regex") != kPattern || steps[0].at("behavior") != "Isolated" ||
        steps[0].at("invert") != false || steps[1].at("type") != "ByteLevel" ||
        steps[1].at("add_prefix_space") != false || steps[1].at("use_regex") != false)
        throw std::runtime_error("glm_ocr tokenizer.json pre-tokenizer differs from the family's");
    if (json.at("decoder").at("type") != "ByteLevel")
        throw std::runtime_error("glm_ocr tokenizer.json must decode byte-level");
    ignore_merges_ = model.value("ignore_merges", false);

    // Every id is checked before it sizes or indexes id_to_token_.
    const auto token_id = [](const nlohmann::json& value) {
        const auto id = value.get<std::int64_t>();
        if (id < 0 || id >= std::numeric_limits<std::int32_t>::max())
            throw std::runtime_error("glm_ocr tokenizer.json has a token id out of range");
        return static_cast<std::int32_t>(id);
    };
    const auto& vocab = model.at("vocab");
    std::size_t size = 0;
    for (auto it = vocab.begin(); it != vocab.end(); ++it)
        size = std::max<std::size_t>(size, static_cast<std::size_t>(token_id(it.value())) + 1);
    for (const auto& added : json.at("added_tokens"))
        size = std::max<std::size_t>(size, static_cast<std::size_t>(token_id(added.at("id"))) + 1);
    id_to_token_.resize(size);
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const auto id = token_id(it.value());
        vocab_.emplace(it.key(), id);
        id_to_token_[static_cast<std::size_t>(id)] = it.key();
    }
    std::int32_t rank = 0;
    for (const auto& merge : model.at("merges")) {
        std::string key;
        if (merge.is_array()) {
            key = merge.at(0).get<std::string>() + " " + merge.at(1).get<std::string>();
        } else {
            key = merge.get<std::string>();
        }
        merge_rank_.emplace(std::move(key), rank++);
    }
    for (const auto& added : json.at("added_tokens")) {
        if (added.value("lstrip", false) || added.value("rstrip", false) ||
            added.value("single_word", false) || added.value("normalized", false))
            throw std::runtime_error("glm_ocr tokenizer.json has an added token it cannot match");
        Added token{added.at("content").get<std::string>(), token_id(added.at("id")),
                    added.value("special", false)};
        if (token.content.empty())
            throw std::runtime_error("glm_ocr tokenizer.json has an empty added token");
        added_by_id_.emplace(token.id, added_.size());
        id_to_token_[static_cast<std::size_t>(token.id)] = token.content;
        added_.push_back(std::move(token));
    }

    // GPT-2 byte-to-unicode table: printable bytes map to themselves, the rest to 256 + n.
    std::uint32_t next = 256;
    for (std::uint32_t b = 0; b < 256; ++b) {
        const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
        const std::uint32_t cp = printable ? b : next++;
        utf8_append(byte_to_unicode_[b], cp);
        unicode_to_byte_.emplace(cp, static_cast<std::uint8_t>(b));
    }
}

std::int32_t Tokenizer::added_token_id(std::string_view content) const {
    for (const auto& token : added_)
        if (token.content == content)
            return token.id;
    throw std::runtime_error("glm_ocr tokenizer has no added token " + std::string(content));
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<std::int32_t> ids;
    std::size_t start = 0, pos = 0;
    while (pos < text.size()) {
        // Leftmost-longest match of an added token.
        const Added* best = nullptr;
        for (const auto& token : added_) {
            if (text.compare(pos, token.content.size(), token.content) == 0 &&
                (best == nullptr || token.content.size() > best->content.size()))
                best = &token;
        }
        if (best == nullptr) {
            ++pos;
            continue;
        }
        encode_span(text.substr(start, pos - start), ids);
        ids.push_back(best->id);
        pos += best->content.size();
        start = pos;
    }
    encode_span(text.substr(start), ids);
    return ids;
}

void Tokenizer::encode_span(std::string_view text, std::vector<std::int32_t>& ids) const {
    if (text.empty())
        return;
    for (const auto piece : pre_tokenize(text))
        encode_piece(piece, ids);
}

void Tokenizer::encode_piece(std::string_view piece, std::vector<std::int32_t>& ids) const {
    const auto mapped = byte_level(piece, byte_to_unicode_);
    if (ignore_merges_) {
        const auto whole = vocab_.find(mapped);
        if (whole != vocab_.end()) {
            ids.push_back(whole->second);
            return;
        }
    }
    std::vector<std::string> symbols;
    for (const char ch : piece)
        symbols.push_back(byte_to_unicode_[static_cast<std::uint8_t>(ch)]);
    while (symbols.size() > 1) {
        std::int32_t best_rank = std::numeric_limits<std::int32_t>::max();
        std::size_t best = symbols.size();
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = merge_rank_.find(symbols[i] + " " + symbols[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best == symbols.size())
            break;
        // Merge every non-overlapping occurrence of the best pair, left to right.
        const std::string left = symbols[best], right = symbols[best + 1];
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (std::size_t i = 0; i < symbols.size(); ++i) {
            if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {
                merged.push_back(left + right);
                ++i;
            } else {
                merged.push_back(symbols[i]);
            }
        }
        symbols = std::move(merged);
    }
    for (const auto& symbol : symbols) {
        const auto it = vocab_.find(symbol);
        if (it == vocab_.end())
            throw std::runtime_error("glm_ocr tokenizer produced a symbol outside its vocabulary");
        ids.push_back(it->second);
    }
}

std::string Tokenizer::decode(Span<const std::int32_t> ids, bool skip_special) const {
    std::string bytes;
    for (const auto id : ids) {
        // The LM head is wider than tokenizer.json; ids without a token decode to nothing,
        // as in the tokenizers library.
        if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size() ||
            id_to_token_[static_cast<std::size_t>(id)].empty())
            continue;
        const auto added = added_by_id_.find(id);
        if (added != added_by_id_.end()) {
            const auto& token = added_[added->second];
            if (!(token.special && skip_special))
                bytes += token.content;
            continue;
        }
        const auto& token = id_to_token_[static_cast<std::size_t>(id)];
        std::size_t pos = 0;
        while (pos < token.size()) {
            std::uint32_t cp = 0;
            const auto length = utf8_decode(token, pos, cp);
            const auto byte = length == 0 ? unicode_to_byte_.end() : unicode_to_byte_.find(cp);
            if (byte == unicode_to_byte_.end())
                throw std::runtime_error("glm_ocr vocabulary token is not byte-level");
            bytes += static_cast<char>(byte->second);
            pos += length;
        }
    }
    return utf8_lossy(bytes);
}

} // namespace trtmc::glm_ocr
