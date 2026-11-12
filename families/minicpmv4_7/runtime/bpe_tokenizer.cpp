/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Byte-level BPE for the MiniCPM-V 4.7 tokenizer.json, as Hugging Face tokenizers applies it:
// added tokens are matched on the raw text, every other segment is NFC-normalized, split with
// the checkpoint's pattern (Isolated) and byte-level BPE encoded. The schema is validated
// exactly; any other normalizer, pattern, model or added-token contract is rejected.

#include "families/minicpmv4_7/runtime/tokenizer.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/utf8.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc::minicpmv4_7 {
namespace {

constexpr std::int32_t kBaseVocabSize = 248044;
constexpr std::int32_t kFullVocabSize = 248144;
constexpr std::size_t kMergeCount = 247587;
constexpr std::size_t kAddedTokenCount = 100;
constexpr std::size_t kSpecialTokenCount = 88;

// The Split pattern exactly as tokenizer.json declares it.
constexpr std::string_view kPretokenizerPattern =
    R"regex((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)regex";
// The same pattern for ICU. Hugging Face tokenizers evaluates it with Oniguruma, whose \s is the
// Unicode White_Space property; ICU's \s omits U+000B and U+0085, so spell the property out.
constexpr std::string_view kIcuPattern =
    R"regex((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\p{White_Space}\p{L}\p{N}]+[\r\n]*|\p{White_Space}*[\r\n]+|\p{White_Space}+(?!\P{White_Space})|\p{White_Space}+)regex";

[[noreturn]] void schema_error(const std::string& detail) {
    throw std::runtime_error("MiniCPM-V 4.7 tokenizer.json " + detail);
}

void require_schema(bool condition, const std::string& detail) {
    if (!condition)
        schema_error(detail);
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

// GPT-2 byte-level alphabet: printable bytes map to themselves, the rest to U+0100 onward.
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

// Rust String::from_utf8_lossy replacement grouping, as Hugging Face byte-level decoding uses.
std::string utf8_lossy(std::string_view bytes) {
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

// Hugging Face tokenizers only ever sees valid Unicode text, so reject anything else rather than
// substituting replacement characters.
void require_utf8(std::string_view text) {
    if (text.size() > static_cast<std::size_t>(INT32_MAX))
        throw std::invalid_argument("MiniCPM-V 4.7 prompt text is too long");
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(text.data());
    const auto length = static_cast<std::int32_t>(text.size());
    std::int32_t offset = 0;
    while (offset < length) {
        UChar32 codepoint = 0;
        U8_NEXT(bytes, offset, length, codepoint);
        if (codepoint < 0)
            throw std::invalid_argument("MiniCPM-V 4.7 prompt text is not valid UTF-8");
    }
}

// One non-added-token segment: NFC, then the Split pattern with Isolated behavior.
std::vector<std::string> pre_tokenize(std::string_view text) {
    UErrorCode status = U_ZERO_ERROR;
    const auto* nfc = icu::Normalizer2::getNFCInstance(status);
    if (U_FAILURE(status))
        throw std::runtime_error("MiniCPM-V 4.7 cannot initialize NFC normalization");
    icu::UnicodeString normalized;
    nfc->normalize(icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), text.size())),
                   normalized, status);
    if (U_FAILURE(status))
        throw std::invalid_argument("MiniCPM-V 4.7 cannot normalize prompt text");
    static const auto pattern = [] {
        UErrorCode code = U_ZERO_ERROR;
        std::unique_ptr<icu::RegexPattern> value(icu::RegexPattern::compile(
            icu::UnicodeString::fromUTF8(icu::StringPiece(kIcuPattern.data(), kIcuPattern.size())),
            0, code));
        if (U_FAILURE(code))
            throw std::runtime_error("MiniCPM-V 4.7 cannot compile the Split pattern");
        return value;
    }();
    std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(normalized, status));
    std::vector<std::string> result;
    // Isolated: every match is a piece, and so is any unmatched text between matches.
    int32_t previous = 0;
    auto append = [&](int32_t begin, int32_t end) {
        if (end <= begin)
            return;
        std::string piece;
        normalized.tempSubStringBetween(begin, end).toUTF8String(piece);
        result.push_back(std::move(piece));
    };
    while (U_SUCCESS(status) && matcher->find(status)) {
        const int32_t begin = matcher->start(status);
        const int32_t end = matcher->end(status);
        append(previous, begin);
        append(begin, end);
        previous = end;
    }
    if (U_FAILURE(status))
        throw std::invalid_argument("MiniCPM-V 4.7 cannot split prompt text");
    append(previous, normalized.length());
    return result;
}

struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& pair) const {
        const std::size_t first = std::hash<std::string>{}(pair.first);
        const std::size_t second = std::hash<std::string>{}(pair.second);
        return first ^ (second + 0x9e3779b9U + (first << 6U) + (first >> 2U));
    }
};

class BpeTokenizer final : public ITokenizer {
  public:
    static std::unique_ptr<BpeTokenizer> create(const char* data, std::size_t size) {
        if (data == nullptr || size == 0)
            schema_error("is empty");
        auto tokenizer = std::unique_ptr<BpeTokenizer>(new BpeTokenizer());
        tokenizer->parse(data, size);
        return tokenizer;
    }

    std::vector<std::int32_t> encode(const std::string& text) const override {
        require_utf8(text);
        std::vector<std::int32_t> result;
        for (const auto& segment : split_added_tokens(text)) {
            if (segment.added_id >= 0)
                result.push_back(segment.added_id);
            else
                encode_text(segment.text, result);
        }
        return result;
    }

    std::string decode(const std::vector<std::int32_t>& ids) const override {
        std::string encoded;
        for (const std::int32_t id : ids) {
            if (id < 0 || id >= static_cast<std::int32_t>(vocabulary_.size()))
                throw std::invalid_argument("MiniCPM-V 4.7 token ID is outside the vocabulary");
            if (special_ids_.count(id) == 0)
                encoded += vocabulary_[static_cast<std::size_t>(id)];
        }
        std::string bytes;
        std::size_t position = 0;
        const auto& reverse = byte_tables().token_to_byte;
        while (position < encoded.size()) {
            const std::size_t start = position;
            const char32_t codepoint = next_codepoint(encoded, position);
            const auto found = reverse.find(codepoint);
            if (found != reverse.end())
                bytes.push_back(static_cast<char>(found->second));
            else
                bytes.append(encoded, start, position - start);
        }
        return utf8_lossy(bytes);
    }

    std::int32_t id_for_token(std::string_view token) const override {
        const auto found = token_to_id_.find(std::string(token));
        return found == token_to_id_.end() ? -1 : found->second;
    }

  private:
    struct Segment {
        std::string text;
        std::int32_t added_id{-1};
    };

    struct MergeCandidate {
        int rank{INT_MAX};
        std::string first;
        std::string second;
    };

    BpeTokenizer() = default;

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
                       "does not match the Split pattern this runtime implements");
        const auto& byte_level = stages.at(1);
        require_schema(byte_level.is_object() && byte_level.size() == 4 &&
                           byte_level.at("type") == "ByteLevel" &&
                           byte_level.at("add_prefix_space") == false &&
                           byte_level.at("trim_offsets") == true &&
                           byte_level.at("use_regex") == false,
                       "does not match the ByteLevel pre-tokenizer contract");
        const auto& post = root.at("post_processor");
        require_schema(post.is_object() && post.size() == 4 && post.at("type") == "ByteLevel" &&
                           post.at("add_prefix_space") == false &&
                           post.at("trim_offsets") == false && post.at("use_regex") == false,
                       "must use the ByteLevel post-processor (no added ids)");
        const auto& decoder = root.at("decoder");
        require_schema(decoder.is_object() && decoder.size() == 4 &&
                           decoder.at("type") == "ByteLevel" &&
                           decoder.at("add_prefix_space") == true &&
                           decoder.at("trim_offsets") == true && decoder.at("use_regex") == true,
                       "must use the ByteLevel decoder");
        const auto& model = root.at("model");
        require_schema(model.is_object() && model.size() == 10 && model.at("type") == "BPE" &&
                           model.at("dropout").is_null() && model.at("unk_token").is_null() &&
                           model.at("continuing_subword_prefix") == "" &&
                           model.at("end_of_word_suffix") == "" && model.at("fuse_unk") == false &&
                           model.at("byte_fallback") == false && model.at("ignore_merges") == false,
                       "must use a plain byte-level BPE model");
        require_schema(model.at("vocab").is_object() &&
                           model.at("vocab").size() == static_cast<std::size_t>(kBaseVocabSize) &&
                           model.at("merges").is_array() &&
                           model.at("merges").size() == kMergeCount,
                       "has an unexpected BPE vocabulary or merge count");
    }

    void parse(const char* data, std::size_t size) {
        try {
            const auto root = nlohmann::json::parse(data, data + size);
            validate_schema(root);
            parse_vocabulary(root.at("model").at("vocab"));
            parse_added_tokens(root.at("added_tokens"));
            parse_merges(root.at("model").at("merges"));
            validate_protocol_tokens();
        } catch (const nlohmann::json::exception& error) {
            schema_error(std::string("is invalid: ") + error.what());
        }
    }

    void parse_vocabulary(const nlohmann::json& values) {
        vocabulary_.resize(kFullVocabSize);
        token_to_id_.reserve(kFullVocabSize);
        std::vector<bool> seen(kBaseVocabSize);
        for (const auto& [token, value] : values.items()) {
            require_schema(value.is_number_integer(), "contains a non-integer vocabulary ID");
            const auto id = value.get<std::int32_t>();
            require_schema(id >= 0 && id < kBaseVocabSize,
                           "contains an out-of-range vocabulary ID");
            require_schema(!seen[static_cast<std::size_t>(id)],
                           "contains a duplicate vocabulary ID");
            seen[static_cast<std::size_t>(id)] = true;
            vocabulary_[static_cast<std::size_t>(id)] = token;
            token_to_id_.emplace(token, id);
        }
        require_schema(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }),
                       "base vocabulary is not dense");
        for (const auto& token : byte_tables().byte_to_token)
            require_schema(token_to_id_.count(token) == 1,
                           "base vocabulary is missing a byte-level token");
    }

    void parse_added_tokens(const nlohmann::json& values) {
        require_schema(values.is_array() && values.size() == kAddedTokenCount,
                       "has an unexpected added-token count");
        std::size_t special_count = 0;
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto& value = values.at(index);
            require_schema(value.is_object() && value.size() == 7,
                           "contains an invalid added-token entry");
            const std::int32_t id = value.at("id").get<std::int32_t>();
            const std::string content = value.at("content").get<std::string>();
            require_schema(id == kBaseVocabSize + static_cast<std::int32_t>(index) &&
                               !content.empty() && value.at("single_word") == false &&
                               value.at("lstrip") == false && value.at("rstrip") == false &&
                               value.at("normalized") == false,
                           "contains an unsupported added-token contract");
            require_schema(std::all_of(content.begin(), content.end(),
                                       [](unsigned char byte) { return byte < 0x80U; }),
                           "contains a non-ASCII added token");
            vocabulary_[static_cast<std::size_t>(id)] = content;
            require_schema(token_to_id_.emplace(content, id).second,
                           "contains duplicate added-token content");
            if (value.at("special").get<bool>()) {
                special_ids_.insert(id);
                ++special_count;
            }
            added_tokens_.emplace_back(content, id);
        }
        require_schema(special_count == kSpecialTokenCount,
                       "has an unexpected special-token count");
        // Leftmost-longest matching, as the added vocabulary's Aho-Corasick matcher does.
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
            require_schema(merge_rank_.emplace(std::move(pair), static_cast<int>(index)).second,
                           "contains a duplicate BPE merge");
        }
    }

    void validate_protocol_tokens() const {
        const auto require_token = [&](std::string_view token, std::int32_t id, bool special) {
            const auto found = token_to_id_.find(std::string(token));
            require_schema(found != token_to_id_.end() && found->second == id &&
                               (special_ids_.count(id) != 0) == special,
                           "does not hold the chat and image protocol tokens");
        };
        require_token("<|endoftext|>", 248044, true);
        require_token("<|im_start|>", 248045, true);
        require_token("<|im_end|>", 248046, true);
        require_token("<|image_pad|>", 248056, true);
        require_token("<think>", 248068, false);
        require_token("</think>", 248069, false);
        require_token("<image>", 248078, true);
        require_token("</image>", 248079, true);
        require_token("<slice>", 248088, true);
        require_token("</slice>", 248089, true);
        require_token("<image_id>", 248090, true);
        require_token("</image_id>", 248091, true);
    }

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

    MergeCandidate best_merge(const std::vector<std::string>& tokens) const {
        MergeCandidate best;
        for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
            const auto found = merge_rank_.find({tokens[index], tokens[index + 1]});
            if (found != merge_rank_.end() && found->second < best.rank)
                best = {found->second, tokens[index], tokens[index + 1]};
        }
        return best;
    }

    static std::vector<std::string> merge_pair(std::vector<std::string> tokens,
                                               const MergeCandidate& merge) {
        std::vector<std::string> result;
        result.reserve(tokens.size());
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            if (index + 1 < tokens.size() && tokens[index] == merge.first &&
                tokens[index + 1] == merge.second) {
                result.push_back(merge.first + merge.second);
                ++index;
            } else {
                result.push_back(std::move(tokens[index]));
            }
        }
        return result;
    }

    std::vector<std::string> apply_merges(std::vector<std::string> tokens) const {
        while (tokens.size() > 1) {
            const auto merge = best_merge(tokens);
            if (merge.rank == INT_MAX)
                break;
            tokens = merge_pair(std::move(tokens), merge);
        }
        return tokens;
    }

    void encode_text(std::string_view text, std::vector<std::int32_t>& result) const {
        for (const auto& word : pre_tokenize(text)) {
            for (const auto& token : apply_merges(byte_encode(word))) {
                const auto found = token_to_id_.find(token);
                if (found == token_to_id_.end())
                    schema_error("cannot encode a byte-level BPE token");
                result.push_back(found->second);
            }
        }
    }

    std::vector<std::string> vocabulary_;
    std::unordered_map<std::string, std::int32_t> token_to_id_;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_rank_;
    std::unordered_set<std::int32_t> special_ids_;
    std::vector<std::pair<std::string, std::int32_t>> added_tokens_;
};

} // namespace

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const char* tokenizer_json_data,
                                               std::size_t tokenizer_json_size) {
    return BpeTokenizer::create(tokenizer_json_data, tokenizer_json_size);
}

} // namespace trtmc::minicpmv4_7
