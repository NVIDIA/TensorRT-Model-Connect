/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ministral3/runtime/tokenizer.h"

#include "families/ministral3/runtime/unicode_data.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <queue>
#include <stdexcept>

namespace trtmc::ministral3 {
namespace {

using nlohmann::json;
using unicode_data::CharClass;

// The Split pattern of the supported tokenizer.json, as stored there.
constexpr std::string_view kTekkenPattern =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|)"
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|)"
    R"(\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

CharClass char_class(char32_t cp) {
    const auto* begin = std::begin(unicode_data::kClasses);
    const auto* end = std::end(unicode_data::kClasses);
    const auto* found = std::upper_bound(
        begin, end, cp, [](char32_t value, const auto& range) { return value < range.lo; });
    if (found == begin)
        return unicode_data::kOther;
    --found;
    return cp <= found->hi ? found->cls : unicode_data::kOther;
}

bool white_space(char32_t cp) {
    for (const auto& range : unicode_data::kWhiteSpace)
        if (cp >= range.lo && cp <= range.hi)
            return true;
    return false;
}

struct CodePoint {
    char32_t value;
    std::size_t byte;
    CharClass cls;
    bool space;
};

std::vector<CodePoint> decode(std::string_view text, std::size_t begin, std::size_t end) {
    std::vector<CodePoint> out;
    out.reserve(end - begin);
    std::size_t pos = begin;
    while (pos < end) {
        const auto lead = static_cast<unsigned char>(text[pos]);
        std::size_t length = 1;
        char32_t value = lead;
        char32_t minimum = 0;
        if ((lead & 0xE0) == 0xC0) {
            length = 2;
            value = lead & 0x1F;
            minimum = 0x80;
        } else if ((lead & 0xF0) == 0xE0) {
            length = 3;
            value = lead & 0x0F;
            minimum = 0x800;
        } else if ((lead & 0xF8) == 0xF0) {
            length = 4;
            value = lead & 0x07;
            minimum = 0x10000;
        } else if (lead >= 0x80) {
            throw std::invalid_argument("ministral3 text is not valid UTF-8");
        }
        if (end - pos < length)
            throw std::invalid_argument("ministral3 text is not valid UTF-8");
        for (std::size_t index = 1; index < length; ++index) {
            const auto byte = static_cast<unsigned char>(text[pos + index]);
            if ((byte & 0xC0) != 0x80)
                throw std::invalid_argument("ministral3 text is not valid UTF-8");
            value = (value << 6) | (byte & 0x3F);
        }
        if (value < minimum || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF))
            throw std::invalid_argument("ministral3 text is not valid UTF-8");
        out.push_back({value, pos, char_class(value), white_space(value)});
        pos += length;
    }
    return out;
}

// Leftmost-first matcher of the seven alternatives of kTekkenPattern with
// Oniguruma's greedy backtracking order; each function returns the match end.
class Matcher {
  public:
    explicit Matcher(const std::vector<CodePoint>& points) : p_(points), n_(points.size()) {}

    std::size_t match(std::size_t i) const {
        for (auto alternative :
             {&Matcher::letters_lower, &Matcher::letters_upper, &Matcher::number, &Matcher::other,
              &Matcher::newlines, &Matcher::spaces_before_space, &Matcher::spaces})
            if (const auto end = (this->*alternative)(i))
                return *end;
        return i;
    }

  private:
    bool upper(std::size_t i) const {
        const auto c = p_[i].cls;
        return c == unicode_data::kUpper || c == unicode_data::kTitle ||
               c == unicode_data::kModifier || c == unicode_data::kOtherLetter ||
               c == unicode_data::kMark;
    }
    bool lower(std::size_t i) const {
        const auto c = p_[i].cls;
        return c == unicode_data::kLower || c == unicode_data::kModifier ||
               c == unicode_data::kOtherLetter || c == unicode_data::kMark;
    }
    bool letter(std::size_t i) const {
        const auto c = p_[i].cls;
        return c != unicode_data::kOther && c != unicode_data::kMark && c != unicode_data::kNumber;
    }
    bool digit(std::size_t i) const { return p_[i].cls == unicode_data::kNumber; }
    bool newline(std::size_t i) const { return p_[i].value == '\n' || p_[i].value == '\r'; }
    // [^\r\n\p{L}\p{N}]
    bool prefix(std::size_t i) const { return !newline(i) && !letter(i) && !digit(i); }
    // [^\s\p{L}\p{N}]
    bool symbol(std::size_t i) const { return !p_[i].space && !letter(i) && !digit(i); }

    std::size_t run(std::size_t i, bool (Matcher::*member)(std::size_t) const) const {
        while (i < n_ && (this->*member)(i))
            ++i;
        return i;
    }

    // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
    std::optional<std::size_t> letters_lower(std::size_t i) const {
        for (std::size_t take = prefix(i) ? 1 : 0;; --take) {
            const auto j = i + take;
            const auto upper_end = run(j, &Matcher::upper);
            for (auto split = upper_end;; --split) {
                const auto end = run(split, &Matcher::lower);
                if (end > split)
                    return end;
                if (split == j)
                    break;
            }
            if (take == 0)
                return std::nullopt;
        }
    }
    // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
    std::optional<std::size_t> letters_upper(std::size_t i) const {
        for (std::size_t take = prefix(i) ? 1 : 0;; --take) {
            const auto j = i + take;
            const auto upper_end = run(j, &Matcher::upper);
            if (upper_end > j)
                return run(upper_end, &Matcher::lower);
            if (take == 0)
                return std::nullopt;
        }
    }
    // \p{N}
    std::optional<std::size_t> number(std::size_t i) const {
        return digit(i) ? std::optional<std::size_t>(i + 1) : std::nullopt;
    }
    //  ?[^\s\p{L}\p{N}]+[\r\n/]*
    std::optional<std::size_t> other(std::size_t i) const {
        for (std::size_t take = p_[i].value == ' ' ? 1 : 0;; --take) {
            const auto j = i + take;
            const auto end = run(j, &Matcher::symbol);
            if (end > j) {
                auto tail = end;
                while (tail < n_ && (newline(tail) || p_[tail].value == '/'))
                    ++tail;
                return tail;
            }
            if (take == 0)
                return std::nullopt;
        }
    }
    bool is_space(std::size_t i) const { return p_[i].space; }
    // \s*[\r\n]+
    std::optional<std::size_t> newlines(std::size_t i) const {
        const auto space_end = run(i, &Matcher::is_space);
        for (auto split = space_end;; --split) {
            const auto end = run(split, &Matcher::newline);
            if (end > split)
                return end;
            if (split == i)
                return std::nullopt;
        }
    }
    // \s+(?!\S)
    std::optional<std::size_t> spaces_before_space(std::size_t i) const {
        const auto space_end = run(i, &Matcher::is_space);
        for (auto end = space_end; end > i; --end)
            if (end == n_ || p_[end].space)
                return end;
        return std::nullopt;
    }
    // \s+
    std::optional<std::size_t> spaces(std::size_t i) const {
        const auto end = run(i, &Matcher::is_space);
        return end > i ? std::optional<std::size_t>(end) : std::nullopt;
    }

    const std::vector<CodePoint>& p_;
    std::size_t n_;
};

std::vector<std::pair<std::size_t, std::size_t>>
pre_tokenize_range(std::string_view text, std::size_t begin, std::size_t end) {
    const auto points = decode(text, begin, end);
    const Matcher matcher(points);
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    auto byte_at = [&](std::size_t index) {
        return index < points.size() ? points[index].byte : end;
    };
    std::size_t index = 0;
    std::optional<std::size_t> gap;
    while (index < points.size()) {
        const auto match_end = matcher.match(index);
        if (match_end == index) {
            if (!gap)
                gap = index;
            ++index;
            continue;
        }
        if (gap) {
            spans.emplace_back(byte_at(*gap), byte_at(index));
            gap.reset();
        }
        spans.emplace_back(byte_at(index), byte_at(match_end));
        index = match_end;
    }
    if (gap)
        spans.emplace_back(byte_at(*gap), end);
    return spans;
}

std::string utf8(char32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

// The GPT-2 byte-to-character alphabet of the ByteLevel pre-tokenizer.
std::vector<std::string> byte_alphabet() {
    std::vector<std::string> out(256);
    int shifted = 0;
    for (int byte = 0; byte < 256; ++byte) {
        const bool printable =
            (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        out[static_cast<std::size_t>(byte)] =
            utf8(printable ? static_cast<char32_t>(byte) : static_cast<char32_t>(256 + shifted++));
    }
    return out;
}

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(std::string("ministral3 tokenizer.json: ") + message);
}

void validate_contract(const json& root) {
    require(root.is_object(), "root must be an object");
    require(root.value("normalizer", json()).is_null(), "normalizer must be null");
    require(root.value("truncation", json()).is_null(), "truncation must be null");
    require(root.value("padding", json()).is_null(), "padding must be null");
    json pattern = json::object();
    pattern["Regex"] = std::string(kTekkenPattern);
    json split = json::object();
    split["type"] = "Split";
    split["pattern"] = pattern;
    split["behavior"] = "Isolated";
    split["invert"] = false;
    json byte_level = json::object();
    byte_level["type"] = "ByteLevel";
    byte_level["add_prefix_space"] = false;
    byte_level["trim_offsets"] = true;
    byte_level["use_regex"] = false;
    json pre_tokenizer = json::object();
    pre_tokenizer["type"] = "Sequence";
    pre_tokenizer["pretokenizers"] = json::array();
    pre_tokenizer["pretokenizers"].push_back(split);
    pre_tokenizer["pretokenizers"].push_back(byte_level);
    require(root.value("pre_tokenizer", json()) == pre_tokenizer,
            "pre_tokenizer must be the Tekken Split regex followed by ByteLevel");
    json sequence = json::object();
    sequence["Sequence"] = json::object();
    sequence["Sequence"]["id"] = "A";
    sequence["Sequence"]["type_id"] = 0;
    json single = json::array();
    single.push_back(sequence);
    const auto post = root.find("post_processor");
    require(post != root.end() && post->is_object() &&
                post->value("type", "") == "TemplateProcessing" &&
                post->value("single", json()) == single &&
                post->value("special_tokens", json()) == json::object(),
            "post_processor must be a TemplateProcessing that adds no tokens");
    const auto found = root.find("model");
    require(found != root.end() && found->is_object() && found->value("type", "") == "BPE",
            "model must be BPE");
    const auto& model = *found;
    require(model.value("dropout", json()).is_null(), "BPE dropout must be null");
    require(model.value("unk_token", json()).is_null(), "BPE unk_token must be null");
    require(model.value("continuing_subword_prefix", json()).is_null(),
            "BPE continuing_subword_prefix must be null");
    require(model.value("end_of_word_suffix", json()).is_null(),
            "BPE end_of_word_suffix must be null");
    require(model.value("fuse_unk", true) == false, "BPE fuse_unk must be false");
    require(model.value("byte_fallback", true) == false, "BPE byte_fallback must be false");
    require(model.value("ignore_merges", false) == true, "BPE ignore_merges must be true");
}

std::pair<std::string, std::string> merge_parts(const json& merge) {
    if (merge.is_string()) {
        const auto text = merge.get<std::string>();
        const auto space = text.find(' ');
        require(space != std::string::npos && space > 0 && space + 1 < text.size() &&
                    text.find(' ', space + 1) == std::string::npos,
                "merge strings must be two space-separated tokens");
        return {text.substr(0, space), text.substr(space + 1)};
    }
    require(merge.is_array() && merge.size() == 2 && merge[0].is_string() && merge[1].is_string(),
            "merges must be strings or pairs of strings");
    return {merge[0].get<std::string>(), merge[1].get<std::string>()};
}

std::uint64_t pair_key(std::int32_t left, std::int32_t right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
           static_cast<std::uint32_t>(right);
}

} // namespace

std::vector<std::pair<std::size_t, std::size_t>> tekken_pre_tokenize(std::string_view text) {
    return pre_tokenize_range(text, 0, text.size());
}

std::shared_ptr<const TekkenTokenizer> TekkenTokenizer::load(const char* data, std::size_t size) {
    if (data == nullptr || size == 0)
        throw std::invalid_argument("ministral3 tokenizer.json is empty");
    const auto root = json::parse(data, data + size);
    validate_contract(root);
    std::shared_ptr<TekkenTokenizer> tokenizer(new TekkenTokenizer());

    const auto& vocab = root.at("model").at("vocab");
    require(vocab.is_object() && !vocab.empty(), "BPE vocab must be a nonempty object");
    std::vector<bool> seen(vocab.size(), false);
    for (const auto& [token, value] : vocab.items()) {
        require(value.is_number_integer(), "vocab ids must be integers");
        const auto id = value.get<std::int64_t>();
        require(id >= 0 && static_cast<std::size_t>(id) < vocab.size() &&
                    !seen[static_cast<std::size_t>(id)],
                "vocab ids must be a permutation of 0..size-1");
        seen[static_cast<std::size_t>(id)] = true;
        tokenizer->vocab_.emplace(token, static_cast<std::int32_t>(id));
        // A token produced by BPE spells one byte per byte-level character.
        std::size_t characters = 0;
        for (const auto byte : token)
            characters += (static_cast<unsigned char>(byte) & 0xC0) != 0x80 ? 1 : 0;
        tokenizer->max_token_bytes_ = std::max(tokenizer->max_token_bytes_, characters);
    }
    tokenizer->vocab_size_ = static_cast<std::int64_t>(vocab.size());

    const auto alphabet = byte_alphabet();
    for (std::size_t byte = 0; byte < 256; ++byte) {
        const auto found = tokenizer->vocab_.find(alphabet[byte]);
        require(found != tokenizer->vocab_.end(), "vocab must contain every byte-level symbol");
        tokenizer->byte_ids_[byte] = found->second;
    }

    const auto& merges = root.at("model").at("merges");
    require(merges.is_array(), "BPE merges must be an array");
    std::uint32_t rank = 0;
    for (const auto& merge : merges) {
        const auto [left, right] = merge_parts(merge);
        const auto left_id = tokenizer->vocab_.find(left);
        const auto right_id = tokenizer->vocab_.find(right);
        const auto merged = tokenizer->vocab_.find(left + right);
        require(left_id != tokenizer->vocab_.end() && right_id != tokenizer->vocab_.end() &&
                    merged != tokenizer->vocab_.end(),
                "every merge and its result must be in the vocab");
        const auto inserted = tokenizer->merges_.emplace(
            pair_key(left_id->second, right_id->second), Merge{rank++, merged->second});
        require(inserted.second, "merges must not repeat a pair");
    }

    tokenizer->added_trie_.emplace_back();
    const auto& added = root.value("added_tokens", json::array());
    require(added.is_array(), "added_tokens must be an array");
    for (const auto& token : added) {
        require(token.is_object() && token.value("single_word", true) == false &&
                    token.value("lstrip", true) == false && token.value("rstrip", true) == false &&
                    token.value("normalized", true) == false,
                "added tokens must not strip, match single words or be normalized");
        const auto content = token.at("content").get<std::string>();
        const auto id = token.at("id").get<std::int64_t>();
        require(!content.empty() && id >= 0 && id < tokenizer->vocab_size_,
                "added tokens need content and an id inside the vocab");
        const auto in_vocab = tokenizer->vocab_.find(content);
        require(in_vocab != tokenizer->vocab_.end() && in_vocab->second == id,
                "added token content must map to its id in the vocab");
        // An added token matches its content's UTF-8 bytes.
        tokenizer->max_token_bytes_ = std::max(tokenizer->max_token_bytes_, content.size());
        std::int32_t node = 0;
        for (const auto byte : content) {
            auto& next = tokenizer->added_trie_[static_cast<std::size_t>(node)].next;
            const auto key = static_cast<unsigned char>(byte);
            const auto found = next.find(key);
            if (found != next.end()) {
                node = found->second;
                continue;
            }
            const auto created = static_cast<std::int32_t>(tokenizer->added_trie_.size());
            tokenizer->added_trie_[static_cast<std::size_t>(node)].next.emplace(key, created);
            tokenizer->added_trie_.emplace_back();
            node = created;
        }
        tokenizer->added_trie_[static_cast<std::size_t>(node)].token =
            static_cast<std::int32_t>(id);
    }
    return tokenizer;
}

void TekkenTokenizer::encode_piece(std::string_view text, std::size_t begin, std::size_t end,
                                   std::vector<Token>& out) const {
    static const auto alphabet = byte_alphabet();
    std::string mapped;
    for (auto index = begin; index < end; ++index)
        mapped += alphabet[static_cast<unsigned char>(text[index])];
    if (const auto whole = vocab_.find(mapped); whole != vocab_.end()) {
        out.push_back({whole->second, begin, end});
        return;
    }
    struct Symbol {
        std::int32_t id;
        std::int64_t prev, next;
        std::size_t length;
    };
    struct Candidate {
        std::uint32_t rank;
        std::size_t position;
        std::int32_t id;
        bool operator>(const Candidate& other) const {
            return rank != other.rank ? rank > other.rank : position > other.position;
        }
    };
    const auto count = end - begin;
    std::vector<Symbol> symbols(count);
    for (std::size_t index = 0; index < count; ++index)
        symbols[index] = {byte_ids_[static_cast<unsigned char>(text[begin + index])],
                          static_cast<std::int64_t>(index) - 1,
                          static_cast<std::int64_t>(index) + 1, 1};
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;
    auto push = [&](std::size_t position, std::int32_t left, std::int32_t right) {
        if (const auto found = merges_.find(pair_key(left, right)); found != merges_.end())
            queue.push({found->second.rank, position, found->second.id});
    };
    for (std::size_t index = 0; index + 1 < count; ++index)
        push(index, symbols[index].id, symbols[index + 1].id);
    while (!queue.empty()) {
        const auto top = queue.top();
        queue.pop();
        auto& current = symbols[top.position];
        if (current.length == 0 || current.next < 0 ||
            current.next >= static_cast<std::int64_t>(count))
            continue;
        const auto right_index = static_cast<std::size_t>(current.next);
        const auto right = symbols[right_index];
        const auto still = merges_.find(pair_key(current.id, right.id));
        if (still == merges_.end() || still->second.id != top.id)
            continue;
        current.id = top.id;
        current.length += right.length;
        current.next = right.next;
        symbols[right_index].length = 0;
        if (right.next >= 0 && right.next < static_cast<std::int64_t>(count))
            symbols[static_cast<std::size_t>(right.next)].prev =
                static_cast<std::int64_t>(top.position);
        if (current.prev >= 0)
            push(static_cast<std::size_t>(current.prev),
                 symbols[static_cast<std::size_t>(current.prev)].id, current.id);
        if (current.next >= 0 && current.next < static_cast<std::int64_t>(count))
            push(top.position, current.id, symbols[static_cast<std::size_t>(current.next)].id);
    }
    auto offset = begin;
    for (const auto& symbol : symbols) {
        if (symbol.length == 0)
            continue;
        out.push_back({symbol.id, offset, offset + symbol.length});
        offset += symbol.length;
    }
}

void TekkenTokenizer::encode_segment(std::string_view text, std::size_t begin, std::size_t end,
                                     std::vector<Token>& out) const {
    for (const auto& [piece_begin, piece_end] : pre_tokenize_range(text, begin, end))
        encode_piece(text, piece_begin, piece_end, out);
}

std::vector<Token> TekkenTokenizer::encode(std::string_view text) const {
    std::vector<Token> out;
    std::size_t segment = 0;
    std::size_t position = 0;
    while (position < text.size()) {
        std::int32_t node = 0;
        std::int32_t token = -1;
        std::size_t length = 0;
        for (auto index = position; index < text.size(); ++index) {
            const auto& next = added_trie_[static_cast<std::size_t>(node)].next;
            const auto found = next.find(static_cast<unsigned char>(text[index]));
            if (found == next.end())
                break;
            node = found->second;
            if (added_trie_[static_cast<std::size_t>(node)].token >= 0) {
                token = added_trie_[static_cast<std::size_t>(node)].token;
                length = index + 1 - position;
            }
        }
        if (token < 0) {
            ++position;
            continue;
        }
        encode_segment(text, segment, position, out);
        out.push_back({token, position, position + length});
        position += length;
        segment = position;
    }
    encode_segment(text, segment, text.size(), out);
    return out;
}

} // namespace trtmc::ministral3
