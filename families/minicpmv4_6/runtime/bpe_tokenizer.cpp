/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpmv4_6/runtime/metadata.h"
#include "families/minicpmv4_6/runtime/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/uniset.h>
#include <unicode/utf8.h>
#include <unicode/uvernum.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc::minicpmv4_6 {
namespace {

constexpr std::string_view kPretokenizerPattern =
    R"regex((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)regex";

[[noreturn]] void schema_error(const std::string& detail) {
    throw std::runtime_error("MiniCPM-V 4.6 tokenizer.json " + detail);
}

void require_schema(bool condition, const std::string& detail) {
    if (!condition)
        schema_error(detail);
}

// A token id from JSON: an integer (not a boolean or a fraction) in [0, limit),
// checked before it is narrowed.
std::int32_t token_id(const nlohmann::json& value, std::int64_t limit, const char* what) {
    require_schema(value.is_number_integer(), std::string("contains a non-integer ") + what);
    const bool in_range = value.is_number_unsigned()
                              ? value.get<std::uint64_t>() < static_cast<std::uint64_t>(limit)
                              : value.get<std::int64_t>() >= 0 && value.get<std::int64_t>() < limit;
    require_schema(in_range, std::string("contains an out-of-range ") + what);
    return static_cast<std::int32_t>(value.get<std::int64_t>());
}

std::string codepoint_to_utf8(char32_t codepoint) {
    std::string result;
    if (codepoint <= 0x7F) {
        result.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        result.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        result.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
    return result;
}

char32_t next_codepoint(std::string_view text, std::size_t& position) {
    const auto first = static_cast<unsigned char>(text[position++]);
    if (first < 0x80)
        return first;

    int continuation_count = 0;
    char32_t codepoint = 0;
    if ((first & 0xE0) == 0xC0) {
        continuation_count = 1;
        codepoint = first & 0x1F;
    } else if ((first & 0xF0) == 0xE0) {
        continuation_count = 2;
        codepoint = first & 0x0F;
    } else if ((first & 0xF8) == 0xF0) {
        continuation_count = 3;
        codepoint = first & 0x07;
    } else {
        schema_error("contains an invalid byte-level vocabulary token");
    }

    if (position + static_cast<std::size_t>(continuation_count) > text.size())
        schema_error("contains a truncated byte-level vocabulary token");
    for (int index = 0; index < continuation_count; ++index) {
        const auto next = static_cast<unsigned char>(text[position++]);
        if ((next & 0xC0) != 0x80)
            schema_error("contains an invalid byte-level vocabulary token");
        codepoint = (codepoint << 6) | (next & 0x3F);
    }
    return codepoint;
}

struct ByteEncoderTables {
    std::array<std::string, 256> byte_to_token;
    std::unordered_map<char32_t, std::uint8_t> token_to_byte;

    ByteEncoderTables() {
        std::array<bool, 256> direct{};
        for (int byte = 33; byte <= 126; ++byte)
            direct[static_cast<std::size_t>(byte)] = true;
        for (int byte = 161; byte <= 172; ++byte)
            direct[static_cast<std::size_t>(byte)] = true;
        for (int byte = 174; byte <= 255; ++byte)
            direct[static_cast<std::size_t>(byte)] = true;

        int displaced = 0;
        for (int byte = 0; byte < 256; ++byte) {
            const char32_t codepoint = direct[static_cast<std::size_t>(byte)]
                                           ? static_cast<char32_t>(byte)
                                           : static_cast<char32_t>(256 + displaced++);
            byte_to_token[static_cast<std::size_t>(byte)] = codepoint_to_utf8(codepoint);
            token_to_byte.emplace(codepoint, static_cast<std::uint8_t>(byte));
        }
    }
};

const ByteEncoderTables& byte_tables() {
    static const ByteEncoderTables tables;
    return tables;
}

std::string lossy_utf8(std::string_view bytes) {
    constexpr std::string_view replacement = "\xEF\xBF\xBD";
    const auto continuation = [](unsigned char byte) { return (byte & 0xC0U) == 0x80U; };
    std::string result;
    result.reserve(bytes.size());

    std::size_t position = 0;
    while (position < bytes.size()) {
        const auto first = static_cast<unsigned char>(bytes[position]);
        if (first < 0x80U) {
            result.push_back(bytes[position++]);
            continue;
        }

        std::size_t length = 0;
        if (first >= 0xC2U && first <= 0xDFU) {
            length = 2;
        } else if (first >= 0xE0U && first <= 0xEFU) {
            length = 3;
        } else if (first >= 0xF0U && first <= 0xF4U) {
            length = 4;
        } else {
            result += replacement;
            ++position;
            continue;
        }

        if (position + 1 >= bytes.size()) {
            result += replacement;
            break;
        }
        const auto second = static_cast<unsigned char>(bytes[position + 1]);
        const bool second_valid = continuation(second) && !(first == 0xE0U && second < 0xA0U) &&
                                  !(first == 0xEDU && second >= 0xA0U) &&
                                  !(first == 0xF0U && second < 0x90U) &&
                                  !(first == 0xF4U && second >= 0x90U);
        if (!second_valid) {
            result += replacement;
            ++position;
            continue;
        }

        if (length >= 3) {
            if (position + 2 >= bytes.size()) {
                result += replacement;
                break;
            }
            if (!continuation(static_cast<unsigned char>(bytes[position + 2]))) {
                result += replacement;
                position += 2;
                continue;
            }
        }
        if (length == 4) {
            if (position + 3 >= bytes.size()) {
                result += replacement;
                break;
            }
            if (!continuation(static_cast<unsigned char>(bytes[position + 3]))) {
                result += replacement;
                position += 3;
                continue;
            }
        }

        result.append(bytes.substr(position, length));
        position += length;
    }
    return result;
}

// tokenizers normalizes with the Unicode 9.0 tables of its NFC crate and matches the Split pattern
// with Oniguruma's Unicode 16.0 tables; ICU's own tables (Unicode 15.1 in ICU 74) are pinned to
// both. NFC only touches code points that Unicode 9.0 assigns: a later code point is a starter
// without a decomposition in those tables, so it stays and normalization does not cross it. The
// letter, mark and number classes are ICU's up to Unicode 16.0 plus the code points that
// Unicode 16.0 assigns to them after Unicode 15.0; \s is White_Space in both.
static_assert(U_ICU_VERSION_MAJOR_NUM >= 72,
              "the Unicode 16.0 additions below start from ICU 72's Unicode 15.0 tables");
constexpr std::string_view kLetters16 =
    R"(\x{1C89}-\x{1C8A}\x{A7CB}-\x{A7CD}\x{A7DA}-\x{A7DC}\x{105C0}-\x{105F3}\x{10D4A}-\x{10D65})"
    R"(\x{10D6F}-\x{10D85}\x{10EC2}-\x{10EC4}\x{11380}-\x{11389}\x{1138B}\x{1138E})"
    R"(\x{11390}-\x{113B5}\x{113B7}\x{113D1}\x{113D3}\x{11BC0}-\x{11BE0}\x{13460}-\x{143FA})"
    R"(\x{16100}-\x{1611D}\x{16D40}-\x{16D6C}\x{18CFF}\x{1E5D0}-\x{1E5ED}\x{1E5F0})"
    R"(\x{2EBF0}-\x{2EE5D})";
constexpr std::string_view kMarks16 =
    R"(\x{897}\x{10D69}-\x{10D6D}\x{10EFC}\x{113B8}-\x{113C0}\x{113C2}\x{113C5}\x{113C7}-\x{113CA})"
    R"(\x{113CC}-\x{113D0}\x{113D2}\x{113E1}-\x{113E2}\x{11F5A}\x{1611E}-\x{1612F})"
    R"(\x{1E5EE}-\x{1E5EF})";
constexpr std::string_view kNumbers16 =
    R"(\x{10D40}-\x{10D49}\x{116D0}-\x{116E3}\x{11BF0}-\x{11BF9}\x{16130}-\x{16139})"
    R"(\x{16D70}-\x{16D79}\x{1CCF0}-\x{1CCF9}\x{1E5F1}-\x{1E5FA})";

struct Unicode9Nfc {
    icu::UnicodeSet assigned;
    icu::FilteredNormalizer2 normalizer;
    Unicode9Nfc(const icu::Normalizer2& nfc, UErrorCode& status)
        : assigned(icu::UnicodeString(u"[:age=9.0:]"), status), normalizer(nfc, assigned) {
        assigned.freeze();
    }
};

const icu::Normalizer2& unicode9_nfc() {
    static const auto value = [] {
        UErrorCode status = U_ZERO_ERROR;
        const auto* nfc = icu::Normalizer2::getNFCInstance(status);
        if (U_FAILURE(status))
            schema_error("cannot initialize NFC normalization");
        auto pinned = std::make_unique<Unicode9Nfc>(*nfc, status);
        if (U_FAILURE(status))
            schema_error("cannot build the Unicode 9.0 character set");
        return pinned;
    }();
    return value->normalizer;
}

// kPretokenizerPattern with \p{L}, \p{M} and \p{N} replaced by their Unicode 16.0 classes.
std::string unicode16_split_pattern() {
    std::string pattern(kPretokenizerPattern);
    for (const auto& [property, additions] :
         {std::pair{std::string_view("L"), kLetters16}, std::pair{std::string_view("M"), kMarks16},
          std::pair{std::string_view("N"), kNumbers16}}) {
        const std::string name = "\\p{" + std::string(property) + "}";
        const std::string pinned = "[[" + name + "&&\\p{Age=16.0}]" + std::string(additions) + "]";
        for (auto at = pattern.find(name); at != std::string::npos;
             at = pattern.find(name, at + pinned.size()))
            pattern.replace(at, name.size(), pinned);
    }
    return pattern;
}

// Calls `piece` with each pre-tokenized piece of the normalized text, in order, until it returns
// false.
template <typename Callback>
void for_each_piece(std::string_view text, Callback&& piece) {
    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeString normalized;
    unicode9_nfc().normalize(
        icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), text.size())), normalized,
        status);
    if (U_FAILURE(status))
        schema_error("cannot normalize prompt text");

    static const auto pattern = [] {
        UErrorCode code = U_ZERO_ERROR;
        std::unique_ptr<icu::RegexPattern> value(icu::RegexPattern::compile(
            icu::UnicodeString::fromUTF8(unicode16_split_pattern()), 0, code));
        if (U_FAILURE(code))
            schema_error("cannot compile the pinned Unicode Split pattern");
        return value;
    }();
    std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(normalized, status));
    while (U_SUCCESS(status) && matcher->find(status)) {
        std::string value;
        matcher->group(status).toUTF8String(value);
        if (U_FAILURE(status) || !piece(value))
            break;
    }
    if (U_FAILURE(status))
        schema_error("cannot split prompt text");
}

std::vector<std::string> split_text(std::string_view text) {
    std::vector<std::string> result;
    for_each_piece(text, [&](const std::string& piece) {
        result.push_back(piece);
        return true;
    });
    return result;
}

struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& pair) const {
        const std::size_t first = std::hash<std::string>{}(pair.first);
        const std::size_t second = std::hash<std::string>{}(pair.second);
        return first ^ (second + 0x9e3779b9U + (first << 6U) + (first >> 2U));
    }
};

class BpeTokenizer final : public Tokenizer {
  public:
    static std::unique_ptr<BpeTokenizer> create(const char* data, std::size_t size) {
        if (data == nullptr || size == 0)
            schema_error("is empty");
        auto tokenizer = std::unique_ptr<BpeTokenizer>(new BpeTokenizer());
        tokenizer->parse(data, size);
        return tokenizer;
    }

    std::optional<std::vector<std::int32_t>> encode(std::string_view text,
                                                    std::size_t limit) const override {
        const std::size_t most = max_code_points(limit);
        if (code_points(text, most) > most)
            return std::nullopt;
        std::vector<std::int32_t> result;
        for (const auto& segment : split_added_tokens(text)) {
            if (segment.added_id >= 0)
                result.push_back(segment.added_id);
            else
                encode_text(segment.text, limit, result);
            if (result.size() > limit)
                return std::nullopt;
        }
        return result;
    }

    std::size_t max_code_points(std::size_t tokens) const override {
        constexpr std::size_t kLongestComposition = 4;
        const std::size_t per_token = kLongestComposition * longest_token_;
        if (tokens > std::numeric_limits<std::size_t>::max() / per_token)
            return std::numeric_limits<std::size_t>::max();
        return tokens * per_token;
    }

    std::string decode(const std::vector<std::int32_t>& ids) const override {
        std::string bytes;
        const auto& reverse = byte_tables().token_to_byte;
        for (const std::int32_t id : ids) {
            if (id < 0 || id >= static_cast<std::int32_t>(vocabulary_.size()))
                throw std::invalid_argument("MiniCPM-V 4.6 token ID is outside the vocabulary");
            if (special_ids_.count(id) != 0)
                continue;
            // ByteLevel decoding maps each character of a token back to its byte; a token
            // with a character outside the byte alphabet (an added token) keeps its own
            // UTF-8 bytes whole, as Hugging Face tokenizers do.
            const auto& token = vocabulary_[static_cast<std::size_t>(id)];
            std::string mapped;
            bool byte_level = true;
            for (std::size_t position = 0; position < token.size() && byte_level;) {
                const auto found = reverse.find(next_codepoint(token, position));
                byte_level = found != reverse.end();
                if (byte_level)
                    mapped.push_back(static_cast<char>(found->second));
            }
            bytes += byte_level ? mapped : token;
        }
        // Hugging Face tokenizers decode byte-level output with Rust's
        // String::from_utf8_lossy; match its replacement grouping.
        return lossy_utf8(bytes);
    }

    std::int32_t id_for_token(std::string_view token) const override {
        const auto found = token_to_id_.find(std::string(token));
        return found == token_to_id_.end() ? -1 : found->second;
    }

    std::int32_t vocabulary_size() const override {
        return static_cast<std::int32_t>(vocabulary_.size());
    }

  private:
    struct Segment {
        std::string text;
        std::int32_t added_id{-1};
    };

    BpeTokenizer() = default;

    static void validate_byte_level(const nlohmann::json& value, const char* what) {
        require_schema(value.is_object() && value.size() == 4 && value.at("type") == "ByteLevel" &&
                           value.at("add_prefix_space") == false &&
                           value.at("trim_offsets") == false && value.at("use_regex") == false,
                       std::string("must use the pinned ByteLevel ") + what);
    }

    static void validate_schema(const nlohmann::json& root) {
        require_schema(root.is_object() && root.size() == 9, "has an unexpected top-level shape");
        require_schema(root.at("version") == "1.0" && root.at("truncation").is_null() &&
                           root.at("padding").is_null(),
                       "uses unsupported version, truncation, or padding");
        const auto& normalizer = root.at("normalizer");
        require_schema(normalizer.is_object() && normalizer.size() == 1 &&
                           normalizer.at("type") == "NFC",
                       "must use the NFC normalizer");
        const auto& pre = root.at("pre_tokenizer");
        require_schema(pre.is_object() && pre.size() == 2 && pre.at("type") == "Sequence",
                       "must use a Sequence pre-tokenizer");
        const auto& stages = pre.at("pretokenizers");
        require_schema(stages.is_array() && stages.size() == 2,
                       "must contain exactly the Split and ByteLevel stages");
        const auto& split = stages.at(0);
        require_schema(split.is_object() && split.size() == 4 && split.at("type") == "Split" &&
                           split.at("behavior") == "Isolated" && split.at("invert") == false &&
                           split.at("pattern").is_object() && split.at("pattern").size() == 1 &&
                           split.at("pattern").at("Regex") == kPretokenizerPattern,
                       "does not match the pinned Unicode Split pattern");
        validate_byte_level(stages.at(1), "pre-tokenizer");
        validate_byte_level(root.at("post_processor"), "post-processor");
        validate_byte_level(root.at("decoder"), "decoder");
        const auto& model = root.at("model");
        require_schema(model.is_object() && model.size() == 10 && model.at("type") == "BPE" &&
                           model.at("dropout").is_null() && model.at("unk_token").is_null() &&
                           model.at("continuing_subword_prefix") == "" &&
                           model.at("end_of_word_suffix") == "" && model.at("fuse_unk") == false &&
                           model.at("byte_fallback") == false && model.at("ignore_merges") == false,
                       "must use a byte-level BPE model");
        require_schema(model.at("vocab").is_object() && !model.at("vocab").empty() &&
                           model.at("merges").is_array(),
                       "has no BPE vocabulary or merges");
    }

    void parse(const char* data, std::size_t size) {
        nlohmann::json root;
        try {
            root = parse_json_unique(data, size, "tokenizer.json");
            validate_schema(root);
            parse_vocabulary(root.at("model").at("vocab"));
            parse_added_tokens(root.at("added_tokens"));
            parse_merges(root.at("model").at("merges"));
            for (const auto& token : vocabulary_)
                longest_token_ = std::max(longest_token_, code_points(token));
        } catch (const nlohmann::json::exception& error) {
            schema_error(std::string("is invalid: ") + error.what());
        }
    }

    void parse_vocabulary(const nlohmann::json& values) {
        require_schema(values.size() <
                           static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
                       "has too many vocabulary entries");
        base_size_ = static_cast<std::int32_t>(values.size());
        vocabulary_.resize(static_cast<std::size_t>(base_size_));
        token_to_id_.reserve(values.size());
        std::vector<bool> seen(values.size());
        for (const auto& [token, value] : values.items()) {
            const auto id = token_id(value, base_size_, "vocabulary ID");
            require_schema(!seen[static_cast<std::size_t>(id)],
                           "contains a duplicate vocabulary ID");
            seen[static_cast<std::size_t>(id)] = true;
            vocabulary_[static_cast<std::size_t>(id)] = token;
            token_to_id_.emplace(token, id);
        }
        for (const auto& token : byte_tables().byte_to_token)
            require_schema(token_to_id_.count(token) == 1,
                           "base vocabulary is missing a byte-level token");
    }

    void parse_added_tokens(const nlohmann::json& values) {
        require_schema(values.is_array() && !values.empty(), "has no added tokens");
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto& value = values.at(index);
            require_schema(value.is_object() && value.size() == 7,
                           "contains an invalid added-token entry");
            const auto id = token_id(value.at("id"), std::numeric_limits<std::int32_t>::max(),
                                     "added-token ID");
            const auto content = value.at("content").get<std::string>();
            // Added tokens follow the dense base vocabulary and are matched verbatim.
            require_schema(id == base_size_ + static_cast<std::int32_t>(index) &&
                               !content.empty() && value.at("single_word") == false &&
                               value.at("lstrip") == false && value.at("rstrip") == false &&
                               value.at("normalized") == false,
                           "contains an unsupported added-token contract");
            vocabulary_.push_back(content);
            require_schema(token_to_id_.emplace(content, id).second,
                           "contains duplicate added-token content");
            if (value.at("special").get<bool>())
                special_ids_.insert(id);
            added_tokens_.emplace_back(content, id);
        }
        std::stable_sort(added_tokens_.begin(), added_tokens_.end(),
                         [](const auto& left, const auto& right) {
                             return left.first.size() > right.first.size();
                         });
    }

    void parse_merges(const nlohmann::json& values) {
        merge_rank_.reserve(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto& value = values.at(index);
            require_schema(value.is_array() && value.size() == 2 && value.at(0).is_string() &&
                               value.at(1).is_string(),
                           "contains an invalid BPE merge");
            auto pair =
                std::make_pair(value.at(0).get<std::string>(), value.at(1).get<std::string>());
            // Both operands and their concatenation are base-vocabulary tokens.
            for (const auto* token : {&pair.first, &pair.second}) {
                const auto found = token_to_id_.find(*token);
                require_schema(found != token_to_id_.end() && found->second < base_size_,
                               "contains a BPE merge of a token outside the base vocabulary");
            }
            const auto merged = token_to_id_.find(pair.first + pair.second);
            require_schema(merged != token_to_id_.end() && merged->second < base_size_,
                           "contains a BPE merge whose result is outside the base vocabulary");
            require_schema(merge_rank_.emplace(std::move(pair), static_cast<int>(index)).second,
                           "contains a duplicate BPE merge");
        }
    }

    // Leftmost-longest match of the verbatim added tokens, as Hugging Face tokenizers do.
    std::vector<Segment> split_added_tokens(std::string_view text) const {
        std::vector<Segment> result;
        std::size_t position = 0;
        while (position < text.size()) {
            const auto found =
                std::find_if(added_tokens_.begin(), added_tokens_.end(), [&](const auto& token) {
                    return token.first.size() <= text.size() - position &&
                           text.compare(position, token.first.size(), token.first) == 0;
                });
            if (found != added_tokens_.end()) {
                result.push_back({found->first, found->second});
                position += found->first.size();
                continue;
            }
            if (result.empty() || result.back().added_id >= 0)
                result.push_back({});
            result.back().text.push_back(text[position++]);
        }
        return result;
    }

    static std::vector<std::string> byte_encode(std::string_view text) {
        std::vector<std::string> result;
        result.reserve(text.size());
        for (const unsigned char byte : text)
            result.push_back(byte_tables().byte_to_token[byte]);
        return result;
    }

    // Hugging Face tokenizers' BPE: apply the lowest-rank merge at its leftmost position, one
    // occurrence at a time, and re-evaluate the pairs it forms with its neighbors after each
    // merge (a queue entry whose pair has changed since it was queued is skipped).
    std::vector<std::string> apply_merges(std::vector<std::string> symbols) const {
        const std::size_t count = symbols.size();
        if (count < 2)
            return symbols;
        constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> previous(count), next(count);
        std::vector<bool> alive(count, true);
        for (std::size_t i = 0; i < count; ++i) {
            previous[i] = i == 0 ? kNone : i - 1;
            next[i] = i + 1 < count ? i + 1 : kNone;
        }
        struct Entry {
            int rank;
            std::size_t position;
            std::string merged;
        };
        const auto later = [](const Entry& left, const Entry& right) {
            return left.rank != right.rank ? left.rank > right.rank
                                           : left.position > right.position;
        };
        std::priority_queue<Entry, std::vector<Entry>, decltype(later)> queue(later);
        const auto push = [&](std::size_t left) {
            if (left == kNone || next[left] == kNone)
                return;
            const auto found = merge_rank_.find({symbols[left], symbols[next[left]]});
            if (found != merge_rank_.end())
                queue.push({found->second, left, symbols[left] + symbols[next[left]]});
        };
        for (std::size_t i = 0; i + 1 < count; ++i)
            push(i);
        while (!queue.empty()) {
            const Entry top = queue.top();
            queue.pop();
            if (!alive[top.position] || next[top.position] == kNone)
                continue;
            const std::size_t right = next[top.position];
            if (merge_rank_.count({symbols[top.position], symbols[right]}) == 0 ||
                symbols[top.position] + symbols[right] != top.merged)
                continue;
            symbols[top.position] = top.merged;
            alive[right] = false;
            next[top.position] = next[right];
            if (next[right] != kNone)
                previous[next[right]] = top.position;
            push(previous[top.position]);
            push(top.position);
        }
        std::vector<std::string> result;
        for (std::size_t i = 0; i < count; ++i)
            if (alive[i])
                result.push_back(std::move(symbols[i]));
        return result;
    }

    // Appends the ids of the text, piece by piece, until `result` holds more than `limit` ids.
    void encode_text(std::string_view text, std::size_t limit,
                     std::vector<std::int32_t>& result) const {
        for_each_piece(text, [&](const std::string& word) {
            for (const auto& token : apply_merges(byte_encode(word))) {
                const auto found = token_to_id_.find(token);
                if (found == token_to_id_.end())
                    schema_error("cannot encode a byte-level BPE token");
                result.push_back(found->second);
            }
            return result.size() <= limit;
        });
    }

    std::int32_t base_size_{0};
    std::size_t longest_token_{1}; // code points of the longest vocabulary or added token
    std::vector<std::string> vocabulary_;
    std::unordered_map<std::string, std::int32_t> token_to_id_;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_rank_;
    std::unordered_set<std::int32_t> special_ids_;
    std::vector<std::pair<std::string, std::int32_t>> added_tokens_;
};

} // namespace

std::string utf8_lossy(std::string_view bytes) {
    return lossy_utf8(bytes);
}

std::size_t code_points(std::string_view text, std::size_t limit) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(text.data());
    const auto length = static_cast<std::int64_t>(text.size());
    std::size_t count = 0;
    for (std::int64_t position = 0; position < length; ++count) {
        if (count == limit)
            return limit + 1;
        UChar32 value = 0;
        // Advances past one code point, or past the maximal ill-formed subpart that ICU's UTF-8
        // conversion replaces with one U+FFFD.
        U8_NEXT(bytes, position, length, value);
        static_cast<void>(value);
    }
    return count;
}

std::vector<std::string> pre_tokenize(std::string_view text) {
    return split_text(text);
}

std::unique_ptr<Tokenizer> create_tokenizer(const char* tokenizer_json, std::size_t size) {
    return BpeTokenizer::create(tokenizer_json, size);
}

} // namespace trtmc::minicpmv4_6
