/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ipa_tokenizer.cpp — Native C++ IPA tokenizer for MagpieTTS
// =============================================================================
//
// Reimplements NeMo's IPATokenizer in pure C++, eliminating the Python runtime
// dependency for MagpieTTS text tokenization. The tokenizer is entirely
// dictionary-based: phoneme dict lookup for known words, grapheme fallback
// for OOV, and a heteronym set that forces grapheme mode.
//
// Each pronunciation in the dictionary is a string of IPA characters. Each
// individual character (which may be multi-byte UTF-8) maps to a token ID.
// Graphemes are uppercase letters (NeMo default: no prefix).
//
// Data is loaded from 4 bundle sections baked at build time:
//   - magpie_ipa_phoneme_dict: TSV word→pronunciation string
//   - magpie_ipa_heteronyms: one word per line
//   - magpie_ipa_vocab: one token per line (line index = token ID)
//   - magpie_ipa_config: JSON with grapheme_prefix, eos_id, etc.
// =============================================================================

#include "families/magpie_tts/runtime/tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

// Advance past one UTF-8 character and return it as a string.
// Returns empty string at end of input.
std::string next_utf8_char(const std::string& s, std::size_t& pos) {
    if (pos >= s.size())
        return {};
    const auto ch = static_cast<unsigned char>(s[pos]);
    std::size_t len = 1;
    if (ch >= 0xF0)
        len = 4;
    else if (ch >= 0xE0)
        len = 3;
    else if (ch >= 0xC0)
        len = 2;
    if (pos + len > s.size())
        len = s.size() - pos;
    std::string result = s.substr(pos, len);
    pos += len;
    return result;
}

// Split a UTF-8 string into individual characters (each may be multi-byte).
std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    std::size_t pos = 0;
    while (pos < s.size()) {
        chars.push_back(next_utf8_char(s, pos));
    }
    return chars;
}

// ---------------------------------------------------------------------------
// Text preprocessing: curly quote replacement + minimal accent stripping
// ---------------------------------------------------------------------------

struct AccentRange {
    uint32_t first;
    uint32_t last;
    char replacement;
};

constexpr std::array<AccentRange, 16> kLatin1AccentRanges{{
    {0xC0, 0xC5, 'A'},
    {0xC7, 0xC7, 'C'},
    {0xC8, 0xCB, 'E'},
    {0xCC, 0xCF, 'I'},
    {0xD1, 0xD1, 'N'},
    {0xD2, 0xD6, 'O'},
    {0xD9, 0xDC, 'U'},
    {0xDD, 0xDD, 'Y'},
    {0xE0, 0xE5, 'a'},
    {0xE7, 0xE7, 'c'},
    {0xE8, 0xEB, 'e'},
    {0xEC, 0xEF, 'i'},
    {0xF1, 0xF1, 'n'},
    {0xF2, 0xF6, 'o'},
    {0xF9, 0xFC, 'u'},
    {0xFD, 0xFF, 'y'},
}};

char latin1_accent_base(uint32_t codepoint) {
    for (const auto& range : kLatin1AccentRanges) {
        if (codepoint >= range.first && codepoint <= range.last) {
            return range.replacement;
        }
    }
    return '\0';
}

uint32_t decode_two_byte_utf8(unsigned char first, unsigned char second) {
    return (static_cast<uint32_t>(first & 0x1F) << 6) | static_cast<uint32_t>(second & 0x3F);
}

bool try_append_latin1_accent_replacement(const std::string& text, std::size_t pos,
                                          std::string& out, std::size_t& consumed) {
    if (pos + 1 >= text.size()) {
        return false;
    }

    const auto first = static_cast<unsigned char>(text[pos]);
    if (first < 0xC0 || first > 0xC3) {
        return false;
    }

    const auto second = static_cast<unsigned char>(text[pos + 1]);
    const char base = latin1_accent_base(decode_two_byte_utf8(first, second));
    if (base == '\0') {
        return false;
    }

    out.push_back(base);
    consumed = 2;
    return true;
}

char curly_quote_replacement(unsigned char first, unsigned char second, unsigned char third) {
    if (first != 0xE2 || second != 0x80) {
        return '\0';
    }
    if (third == 0x98 || third == 0x99) {
        return '\'';
    }
    if (third == 0x9C || third == 0x9D) {
        return '"';
    }
    return '\0';
}

bool try_append_curly_quote_replacement(const std::string& text, std::size_t pos, std::string& out,
                                        std::size_t& consumed) {
    if (pos + 2 >= text.size()) {
        return false;
    }

    const auto first = static_cast<unsigned char>(text[pos]);
    const auto second = static_cast<unsigned char>(text[pos + 1]);
    const auto third = static_cast<unsigned char>(text[pos + 2]);
    const char replacement = curly_quote_replacement(first, second, third);
    if (replacement == '\0') {
        return false;
    }

    out.push_back(replacement);
    consumed = 3;
    return true;
}

std::string to_lower_ascii(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (char c : text) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

std::string preprocess_text(const std::string& text) {
    std::string out;
    out.reserve(text.size());

    for (std::size_t i = 0; i < text.size();) {
        std::size_t consumed = 0;
        if (try_append_latin1_accent_replacement(text, i, out, consumed) ||
            try_append_curly_quote_replacement(text, i, out, consumed)) {
            i += consumed;
            continue;
        }

        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Word tokenization — hand-rolled 3-state machine
// ---------------------------------------------------------------------------

enum class TokenType { WORD, PIPE_DELIMITED, OTHER };

struct TextToken {
    std::string text;
    TokenType type;
};

bool is_alpha(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool is_word_internal(char c) {
    return is_alpha(c) || c == '-' || c == '\'';
}

std::vector<TextToken> tokenize_text(const std::string& text) {
    std::vector<TextToken> tokens;
    const std::size_t len = text.size();
    std::size_t i = 0;

    while (i < len) {
        if (text[i] == '|') {
            std::size_t end = text.find('|', i + 1);
            if (end != std::string::npos) {
                tokens.push_back({text.substr(i + 1, end - i - 1), TokenType::PIPE_DELIMITED});
                i = end + 1;
                continue;
            }
            tokens.push_back({std::string(1, text[i]), TokenType::OTHER});
            ++i;
            continue;
        }

        if (is_alpha(text[i])) {
            std::size_t start = i;
            ++i;
            while (i < len && is_word_internal(text[i])) {
                ++i;
            }
            while (i > start + 1 && !is_alpha(text[i - 1])) {
                --i;
            }
            tokens.push_back({text.substr(start, i - start), TokenType::WORD});
            continue;
        }

        tokens.push_back({std::string(1, text[i]), TokenType::OTHER});
        ++i;
    }
    return tokens;
}

// ---------------------------------------------------------------------------
// IPA Tokenizer implementation
// ---------------------------------------------------------------------------

class IpaTokenizer final : public ITokenizer {
  public:
    // phoneme_dict: word → list of pronunciation strings (IPA character sequences)
    IpaTokenizer(std::unordered_map<std::string, std::vector<std::string>> phoneme_dict,
                 std::unordered_set<std::string> heteronyms, std::vector<std::string> vocab,
                 std::unordered_map<std::string, int32_t> token2id,
                 std::unordered_set<std::string> known_tokens, std::string grapheme_prefix,
                 int32_t eos_id, bool ignore_ambiguous_words)
        : mPhonemeDict(std::move(phoneme_dict)), mHeteronyms(std::move(heteronyms)),
          mVocab(std::move(vocab)), mToken2Id(std::move(token2id)),
          mKnownTokens(std::move(known_tokens)), mGraphemePrefix(std::move(grapheme_prefix)),
          mEosId(eos_id), mIgnoreAmbiguous(ignore_ambiguous_words) {}

    std::vector<int32_t> encode(const std::string& text) const override {
        const std::string preprocessed = preprocess_text(text);
        const auto text_tokens = tokenize_text(preprocessed);
        auto ipa_tokens = text_tokens_to_ipa_tokens(text_tokens);
        auto filtered = filter_known_tokens(ipa_tokens);
        auto deduped = dedupe_consecutive_spaces(filtered);
        return tokens_to_ids_with_eos(deduped);
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string out;
        for (const int32_t id : ids) {
            if (id == mEosId)
                break;
            if (id >= 0 && static_cast<std::size_t>(id) < mVocab.size()) {
                out += mVocab[static_cast<std::size_t>(id)];
            }
        }
        return out;
    }

    int32_t id_for_token(std::string_view token) const override {
        auto it = mToken2Id.find(std::string(token));
        return (it != mToken2Id.end()) ? it->second : -1;
    }

    std::string token_for_id(int32_t id) const override {
        if (id >= 0 && static_cast<std::size_t>(id) < mVocab.size()) {
            return mVocab[static_cast<std::size_t>(id)];
        }
        return "";
    }

  private:
    std::vector<std::string>
    text_tokens_to_ipa_tokens(const std::vector<TextToken>& text_tokens) const {
        std::vector<std::string> ipa_tokens;
        for (const auto& tok : text_tokens) {
            emit_token_as_ipa(tok, ipa_tokens);
        }
        return ipa_tokens;
    }

    void emit_token_as_ipa(const TextToken& tok, std::vector<std::string>& out) const {
        if (tok.type == TokenType::WORD) {
            g2p_word(tok.text, out);
            return;
        }

        if (tok.type == TokenType::PIPE_DELIMITED) {
            emit_ipa_chars(tok.text, out);
            return;
        }

        for (char c : tok.text) {
            out.emplace_back(1, c);
        }
    }

    std::vector<std::string> filter_known_tokens(const std::vector<std::string>& ipa_tokens) const {
        std::vector<std::string> filtered;
        filtered.reserve(ipa_tokens.size());
        for (const auto& token : ipa_tokens) {
            if (mKnownTokens.count(token) != 0) {
                filtered.push_back(token);
            }
        }
        return filtered;
    }

    static std::vector<std::string>
    dedupe_consecutive_spaces(const std::vector<std::string>& tokens) {
        std::vector<std::string> deduped;
        deduped.reserve(tokens.size());
        for (const auto& token : tokens) {
            if (token == " " && !deduped.empty() && deduped.back() == " ") {
                continue;
            }
            deduped.push_back(token);
        }
        if (!deduped.empty() && deduped.back() == " ") {
            deduped.pop_back();
        }
        return deduped;
    }

    std::vector<int32_t> tokens_to_ids_with_eos(const std::vector<std::string>& tokens) const {
        std::vector<int32_t> ids;
        ids.reserve(tokens.size() + 1);
        for (const auto& token : tokens) {
            const auto it = mToken2Id.find(token);
            if (it != mToken2Id.end()) {
                ids.push_back(it->second);
            }
        }
        ids.push_back(mEosId);
        return ids;
    }

    const std::vector<std::string>* find_pronunciations(const std::string& lower_word) const {
        const auto it = mPhonemeDict.find(lower_word);
        if (it == mPhonemeDict.end()) {
            return nullptr;
        }
        return &it->second;
    }

    bool try_emit_direct_lookup(const std::string& word, const std::string& lower_word,
                                std::vector<std::string>& out) const {
        if (mHeteronyms.count(lower_word) != 0) {
            emit_graphemes(word, out);
            return true;
        }

        const auto* pronunciations = find_pronunciations(lower_word);
        if (pronunciations == nullptr) {
            return false;
        }

        if (pronunciations->size() == 1) {
            emit_ipa_chars((*pronunciations)[0], out);
            return true;
        }

        if (mIgnoreAmbiguous) {
            emit_graphemes(word, out);
            return true;
        }

        emit_ipa_chars((*pronunciations)[0], out);
        return true;
    }

    static bool try_possessive_base(std::string_view lower_word, std::string& base) {
        if (lower_word.size() <= 2 || lower_word.back() != 's') {
            return false;
        }

        if (lower_word[lower_word.size() - 2] == '\'') {
            base.assign(lower_word.data(), lower_word.size() - 2);
            return true;
        }

        base.assign(lower_word.data(), lower_word.size() - 1);
        return true;
    }

    bool try_emit_possessive_fallback(std::string_view lower_word,
                                      std::vector<std::string>& out) const {
        std::string base;
        if (!try_possessive_base(lower_word, base)) {
            return false;
        }

        const auto* pronunciations = find_pronunciations(base);
        if (pronunciations == nullptr || pronunciations->empty()) {
            return false;
        }

        emit_ipa_chars((*pronunciations)[0], out);
        out.emplace_back("z");
        return true;
    }

    // Emit individual UTF-8 characters from an IPA pronunciation string.
    // Each character becomes a separate token (may be multi-byte).
    void emit_ipa_chars(const std::string& pronunciation, std::vector<std::string>& out) const {
        auto chars = utf8_chars(pronunciation);
        for (const auto& ch : chars) {
            out.push_back(ch);
        }
    }

    // G2P for a word token
    void g2p_word(const std::string& word, std::vector<std::string>& out) const {
        const std::string lower = to_lower_ascii(word);
        if (try_emit_direct_lookup(word, lower, out)) {
            return;
        }
        if (try_emit_possessive_fallback(lower, out)) {
            return;
        }
        emit_graphemes(word, out);
    }

    // Emit each alpha character of a word as a grapheme token.
    // NeMo uses uppercase letters as grapheme tokens (no prefix by default).
    // If grapheme_prefix is set (e.g. "#"), emits prefix+lowercase.
    void emit_graphemes(const std::string& word, std::vector<std::string>& out) const {
        for (char c : word) {
            if (!is_alpha(c))
                continue;
            if (mGraphemePrefix.empty()) {
                // NeMo default: uppercase letter as token
                out.emplace_back(1, static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
            } else {
                out.push_back(
                    mGraphemePrefix +
                    std::string(1, static_cast<char>(std::tolower(static_cast<unsigned char>(c)))));
            }
        }
    }

    // word → list of pronunciation strings (each is an IPA character sequence)
    std::unordered_map<std::string, std::vector<std::string>> mPhonemeDict;
    std::unordered_set<std::string> mHeteronyms;
    std::vector<std::string> mVocab;
    std::unordered_map<std::string, int32_t> mToken2Id;
    std::unordered_set<std::string> mKnownTokens;
    std::string mGraphemePrefix;
    int32_t mEosId;
    bool mIgnoreAmbiguous;
};

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

// Parse phoneme dictionary from TSV text:
// word<TAB>pronunciation_string\n
// Each pronunciation is a single IPA string (characters are individual tokens).
// Multiple lines for same word = multiple pronunciations.
std::unordered_map<std::string, std::vector<std::string>> parse_phoneme_dict(const char* data,
                                                                             std::size_t size) {
    std::unordered_map<std::string, std::vector<std::string>> dict;
    const std::string text(data, size);
    std::istringstream iss(text);
    std::string line;

    while (std::getline(iss, line)) {
        if (line.empty())
            continue;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        const auto tab_pos = line.find('\t');
        if (tab_pos == std::string::npos)
            continue;

        std::string word = line.substr(0, tab_pos);
        std::string pronunciation = line.substr(tab_pos + 1);

        if (!pronunciation.empty()) {
            dict[word].push_back(std::move(pronunciation));
        }
    }
    return dict;
}

// Parse heteronyms: one word per line
std::unordered_set<std::string> parse_heteronyms(const char* data, std::size_t size) {
    std::unordered_set<std::string> set;
    const std::string text(data, size);
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (!line.empty())
            set.insert(line);
    }
    return set;
}

// Parse vocabulary: one token per line, line index = token ID
std::vector<std::string> parse_vocab(const char* data, std::size_t size) {
    std::vector<std::string> vocab;
    const std::string text(data, size);
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        vocab.push_back(line);
    }
    return vocab;
}

void validate_ipa_tokenizer_inputs(const char* phoneme_dict_data, std::size_t phoneme_dict_size,
                                   const char* vocab_data, std::size_t vocab_size) {
    if (phoneme_dict_data == nullptr || phoneme_dict_size == 0) {
        throw std::invalid_argument("IPA phoneme dictionary data must not be empty");
    }
    if (vocab_data == nullptr || vocab_size == 0) {
        throw std::invalid_argument("IPA vocabulary data must not be empty");
    }
}

std::unordered_set<std::string> parse_optional_heteronyms(const char* heteronyms_data,
                                                          std::size_t heteronyms_size) {
    if (heteronyms_data == nullptr || heteronyms_size == 0) {
        return {};
    }
    return parse_heteronyms(heteronyms_data, heteronyms_size);
}

struct ParsedIpaTokenizerConfig {
    std::string grapheme_prefix;
    int32_t eos_id{-1};
    bool ignore_ambiguous{false};
};

ParsedIpaTokenizerConfig parse_ipa_tokenizer_config(const std::string& config_text,
                                                    std::size_t vocab_size) {
    const auto config = nlohmann::json::parse(config_text);
    if (!config.is_object())
        throw std::runtime_error("Magpie IPA config must be a JSON object");
    ParsedIpaTokenizerConfig parsed;
    parsed.grapheme_prefix = config.at("grapheme_prefix").get<std::string>();
    parsed.eos_id = config.at("eos_id").get<int32_t>();
    if (parsed.eos_id < 0) {
        parsed.eos_id = static_cast<int32_t>(vocab_size) - 1;
    }
    parsed.ignore_ambiguous = config.at("ignore_ambiguous_words").get<int32_t>() != 0;
    return parsed;
}

struct IpaTokenLookup {
    std::unordered_map<std::string, int32_t> token2id;
    std::unordered_set<std::string> known_tokens;
};

IpaTokenLookup build_ipa_token_lookup(const std::vector<std::string>& vocab) {
    IpaTokenLookup lookup;
    for (std::size_t i = 0; i < vocab.size(); ++i) {
        lookup.token2id.emplace(vocab[i], static_cast<int32_t>(i));
        lookup.known_tokens.insert(vocab[i]);
    }
    return lookup;
}

} // namespace

std::unique_ptr<ITokenizer>
CreateIpaTokenizer(const char* phoneme_dict_data, std::size_t phoneme_dict_size,
                   const char* heteronyms_data, std::size_t heteronyms_size, const char* vocab_data,
                   std::size_t vocab_size, const char* config_data, std::size_t config_size) {
    validate_ipa_tokenizer_inputs(phoneme_dict_data, phoneme_dict_size, vocab_data, vocab_size);

    auto phoneme_dict = parse_phoneme_dict(phoneme_dict_data, phoneme_dict_size);
    auto heteronyms = parse_optional_heteronyms(heteronyms_data, heteronyms_size);
    auto vocab = parse_vocab(vocab_data, vocab_size);

    if (config_data == nullptr || config_size == 0)
        throw std::runtime_error("Magpie IPA config section is empty");
    const std::string config_text(config_data, config_size);
    auto parsed_config = parse_ipa_tokenizer_config(config_text, vocab.size());
    auto token_lookup = build_ipa_token_lookup(vocab);

    return std::make_unique<IpaTokenizer>(std::move(phoneme_dict), std::move(heteronyms),
                                          std::move(vocab), std::move(token_lookup.token2id),
                                          std::move(token_lookup.known_tokens),
                                          std::move(parsed_config.grapheme_prefix),
                                          parsed_config.eos_id, parsed_config.ignore_ambiguous);
}

} // namespace trtmc
