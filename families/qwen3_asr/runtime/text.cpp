/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_asr/runtime/text.h"

#include <cstring>
#include <stdexcept>

namespace trtmc::qwen3_asr {
namespace {

constexpr char kMagic[4] = {'Q', '3', 'A', 'V'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint8_t kUnknown = 2;
constexpr char32_t kReplacement = 0xFFFD;

template <class T>
T read_le(const std::vector<char>& data, std::size_t& offset) {
    if (offset + sizeof(T) > data.size())
        throw std::runtime_error("qwen3_asr: vocabulary table is truncated");
    T value{};
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

// Characters for which Python's str.isspace() is true.
bool is_python_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

std::u32string strip(const std::u32string& text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && is_python_space(text[begin]))
        ++begin;
    while (end > begin && is_python_space(text[end - 1]))
        --end;
    return text.substr(begin, end - begin);
}

std::u32string fix_char_repeats(const std::u32string& s, std::size_t threshold) {
    std::u32string result;
    std::size_t i = 0;
    const std::size_t n = s.size();
    while (i < n) {
        std::size_t count = 1;
        while (i + count < n && s[i + count] == s[i])
            ++count;
        if (count > threshold)
            result.push_back(s[i]);
        else
            result.append(s, i, count);
        i += count;
    }
    return result;
}

std::u32string fix_pattern_repeats(const std::u32string& s, std::size_t threshold,
                                   std::size_t max_length = 20) {
    const std::size_t n = s.size();
    const std::size_t min_repeat_chars = threshold * 2;
    if (n < min_repeat_chars)
        return s;
    std::u32string result;
    std::size_t i = 0;
    bool found = false;
    while (i + min_repeat_chars <= n) {
        found = false;
        for (std::size_t k = 1; k <= max_length; ++k) {
            if (i + k * threshold > n)
                break;
            bool valid = true;
            for (std::size_t rep = 1; rep < threshold && valid; ++rep)
                valid = s.compare(i + rep * k, k, s, i, k) == 0;
            if (!valid)
                continue;
            std::size_t end = i + threshold * k;
            while (end + k <= n && s.compare(end, k, s, i, k) == 0)
                end += k;
            result.append(s, i, k);
            result += fix_pattern_repeats(s.substr(end), threshold, max_length);
            i = n;
            found = true;
            break;
        }
        if (found)
            break;
        result.push_back(s[i]);
        ++i;
    }
    if (!found)
        result.append(s, i, std::u32string::npos);
    return result;
}

std::size_t find(const std::u32string& text, const std::u32string& needle) {
    return needle.empty() ? 0 : text.find(needle);
}

std::u32string to_u32(std::string_view ascii) {
    return {ascii.begin(), ascii.end()};
}

} // namespace

Vocabulary Vocabulary::parse(const std::vector<char>& data) {
    if (data.size() < 12 || std::memcmp(data.data(), kMagic, 4) != 0)
        throw std::runtime_error("qwen3_asr: vocab.bin is not a vocabulary table");
    std::size_t offset = 4;
    if (read_le<std::uint32_t>(data, offset) != kVersion)
        throw std::runtime_error("qwen3_asr: unsupported vocabulary table version");
    const auto count = read_le<std::uint32_t>(data, offset);
    Vocabulary vocabulary;
    vocabulary.pieces_.reserve(count);
    vocabulary.flags_.reserve(count);
    for (std::uint32_t id = 0; id < count; ++id) {
        const auto flag = read_le<std::uint8_t>(data, offset);
        const auto length = read_le<std::uint16_t>(data, offset);
        if (offset + length > data.size() || flag > kUnknown)
            throw std::runtime_error("qwen3_asr: vocabulary table entry is malformed");
        vocabulary.pieces_.emplace_back(data.data() + offset, length);
        vocabulary.flags_.push_back(flag);
        offset += length;
    }
    if (offset != data.size())
        throw std::runtime_error("qwen3_asr: vocabulary table has trailing bytes");
    return vocabulary;
}

std::string Vocabulary::decode(const std::vector<std::int32_t>& token_ids) const {
    std::string bytes;
    for (const auto id : token_ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= pieces_.size())
            throw std::out_of_range("qwen3_asr: token id outside the vocabulary");
        const auto index = static_cast<std::size_t>(id);
        // Special (1) and unknown (2) ids add nothing, as skip_special_tokens=True does.
        if (flags_[index] == 0)
            bytes += pieces_[index];
    }
    return encode_utf8(decode_utf8_replace(bytes));
}

std::u32string decode_utf8_replace(std::string_view bytes) {
    std::u32string out;
    out.reserve(bytes.size());
    std::size_t i = 0;
    const std::size_t n = bytes.size();
    const auto at = [&](std::size_t index) { return static_cast<unsigned char>(bytes[index]); };
    while (i < n) {
        const unsigned char lead = at(i);
        if (lead < 0x80) {
            out.push_back(lead);
            ++i;
            continue;
        }
        std::size_t need = 0;
        unsigned char low = 0x80, high = 0xBF;
        char32_t value = 0;
        if (lead >= 0xC2 && lead <= 0xDF) {
            need = 1;
            value = lead & 0x1F;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            need = 2;
            value = lead & 0x0F;
            if (lead == 0xE0)
                low = 0xA0;
            else if (lead == 0xED)
                high = 0x9F;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            need = 3;
            value = lead & 0x07;
            if (lead == 0xF0)
                low = 0x90;
            else if (lead == 0xF4)
                high = 0x8F;
        } else {
            out.push_back(kReplacement);
            ++i;
            continue;
        }
        // A maximal subpart that stops early becomes one U+FFFD; the offending byte starts over.
        std::size_t consumed = 1;
        bool complete = true;
        for (std::size_t k = 0; k < need; ++k) {
            if (i + consumed >= n) {
                complete = false;
                break;
            }
            const unsigned char next = at(i + consumed);
            const unsigned char lower = k == 0 ? low : 0x80;
            const unsigned char upper = k == 0 ? high : 0xBF;
            if (next < lower || next > upper) {
                complete = false;
                break;
            }
            value = (value << 6) | (next & 0x3F);
            ++consumed;
        }
        out.push_back(complete ? value : kReplacement);
        i += consumed;
    }
    return out;
}

std::string encode_utf8(const std::u32string& text) {
    std::string out;
    out.reserve(text.size());
    for (const char32_t c : text) {
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
        } else if (c < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (c >> 6)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else if (c < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (c >> 12)));
            out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (c >> 18)));
            out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
    }
    return out;
}

std::u32string fix_repetitions(const std::u32string& text, std::int32_t threshold) {
    if (threshold < 1)
        throw std::invalid_argument("qwen3_asr: repetition threshold must be positive");
    const auto limit = static_cast<std::size_t>(threshold);
    return fix_pattern_repeats(fix_char_repeats(text, limit), limit);
}

std::string transcription_from_output(const std::string& decoded, const std::string& marker,
                                      std::int32_t repetition_threshold) {
    std::u32string text = strip(decode_utf8_replace(decoded));
    if (text.empty())
        return {};
    const std::u32string assistant = to_u32("assistant\n");
    if (const auto at = find(text, assistant); at != std::u32string::npos)
        text = text.substr(at + assistant.size());
    text = fix_repetitions(text, repetition_threshold);
    const std::u32string tag = decode_utf8_replace(marker);
    const auto at = find(text, tag);
    if (at == std::u32string::npos)
        return encode_utf8(strip(text));
    return encode_utf8(strip(text.substr(at + tag.size())));
}

} // namespace trtmc::qwen3_asr
