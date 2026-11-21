/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/roberta/runtime/tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

// ─── UTF-8 helpers ───

// Return the byte length of the UTF-8 codepoint starting at s[pos].
inline size_t utf8_char_len(const std::string& s, size_t pos) {
    unsigned char c = static_cast<unsigned char>(s[pos]);
    if (c < 0x80)
        return 1;
    if ((c & 0xE0) == 0xC0)
        return 2;
    if ((c & 0xF0) == 0xE0)
        return 3;
    if ((c & 0xF8) == 0xF0)
        return 4;
    return 1;
}

class PrecompiledNormalizer {
  public:
    explicit PrecompiledNormalizer(const std::string& encoded) {
        const auto bytes = decode_base64(encoded);
        if (bytes.size() <= 4)
            throw std::runtime_error("Invalid Precompiled charsmap header");
        const size_t trie_bytes = read_u32(bytes, 0);
        if (trie_bytes < 1024 || trie_bytes % 1024 != 0 || trie_bytes >= bytes.size() - 4)
            throw std::runtime_error("Invalid Precompiled charsmap trie size");
        mUnits.reserve(trie_bytes / 4);
        for (size_t pos = 4; pos < 4 + trie_bytes; pos += 4)
            mUnits.push_back(read_u32(bytes, pos));
        mReplacements.assign(bytes.begin() + 4 + trie_bytes, bytes.end());
        if (mReplacements.back() != '\0')
            throw std::runtime_error("Invalid Precompiled charsmap string pool");
        if (label(mUnits[0]) != 0 || (mUnits[0] & 256) != 0 || offset(mUnits[0]) == 0)
            throw std::runtime_error("Invalid Precompiled charsmap root");
        for (size_t i = 0; i < mUnits.size(); ++i) {
            const auto unit = mUnits[i];
            if (label(unit) <= 255) {
                const auto child = i ^ offset(unit);
                if ((child | 255) >= mUnits.size())
                    throw std::runtime_error("Invalid Precompiled charsmap trie offset");
                if ((unit & 256) != 0 && (mUnits[child] & 0x80000000U) == 0)
                    throw std::runtime_error("Invalid Precompiled charsmap leaf");
            } else if ((unit & 0x7fffffffU) >= mReplacements.size()) {
                throw std::runtime_error("Invalid Precompiled charsmap replacement offset");
            }
        }
    }

    std::string normalize(const std::string& text) const {
        std::string result;
        result.reserve(text.size());
        for (size_t pos = 0; pos < text.size();) {
            size_t node = offset(mUnits[0]);
            size_t matched = 0;
            size_t replacement = 0;
            for (size_t i = pos; i < text.size(); ++i) {
                const auto byte = static_cast<unsigned char>(text[i]);
                if (byte == 0)
                    break;
                node ^= byte;
                const auto unit = mUnits[node];
                if (label(unit) != byte)
                    break;
                node ^= offset(unit);
                if ((unit & 256) != 0) {
                    matched = i - pos + 1;
                    replacement = mUnits[node] & 0x7fffffffU;
                }
            }
            if (matched != 0) {
                result.append(mReplacements.data() + replacement);
                pos += matched;
            } else {
                const auto length = std::min(utf8_char_len(text, pos), text.size() - pos);
                result.append(text, pos, length);
                pos += length;
            }
        }
        return result;
    }

  private:
    static uint32_t label(uint32_t unit) { return unit & 0x800000ffU; }
    static uint32_t offset(uint32_t unit) { return (unit >> 10) << ((unit & 512) >> 6); }

    static uint32_t read_u32(const std::vector<unsigned char>& bytes, size_t pos) {
        return uint32_t(bytes[pos]) | (uint32_t(bytes[pos + 1]) << 8) |
               (uint32_t(bytes[pos + 2]) << 16) | (uint32_t(bytes[pos + 3]) << 24);
    }

    static std::vector<unsigned char> decode_base64(const std::string& encoded) {
        const std::string alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        if (encoded.empty() || encoded.size() % 4 != 0)
            throw std::runtime_error("Invalid Precompiled charsmap base64 length");
        std::vector<unsigned char> bytes;
        bytes.reserve(encoded.size() / 4 * 3);
        for (size_t i = 0; i < encoded.size(); i += 4) {
            uint32_t value = 0;
            size_t padding = 0;
            for (size_t j = 0; j < 4; ++j) {
                const auto character = encoded[i + j];
                value <<= 6;
                if (character == '=') {
                    if (i + 4 != encoded.size() || j < 2)
                        throw std::runtime_error("Invalid Precompiled charsmap base64 padding");
                    ++padding;
                } else {
                    const auto digit = alphabet.find(character);
                    if (padding != 0 || digit == std::string::npos)
                        throw std::runtime_error("Invalid Precompiled charsmap base64 character");
                    value |= static_cast<uint32_t>(digit);
                }
            }
            if ((padding == 1 && (value & 255) != 0) || (padding == 2 && (value & 65535) != 0))
                throw std::runtime_error("Invalid Precompiled charsmap base64 trailing bits");
            bytes.push_back(static_cast<unsigned char>(value >> 16));
            if (padding < 2)
                bytes.push_back(static_cast<unsigned char>(value >> 8));
            if (padding == 0)
                bytes.push_back(static_cast<unsigned char>(value));
        }
        return bytes;
    }

    std::vector<uint32_t> mUnits;
    std::string mReplacements;
};

std::string lowercase_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// ─── Metaspace Pre-tokenizer ───
// Replaces spaces with ▁ (U+2581) and optionally adds prefix space.

static const std::string kMetaspaceChar = "\xe2\x96\x81"; // ▁ U+2581

inline bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::vector<std::string> whitespace_split(const std::string& text) {
    std::vector<std::string> words;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && is_ws(text[i]))
            ++i;
        if (i >= text.size())
            break;
        size_t start = i;
        while (i < text.size() && !is_ws(text[i]))
            ++i;
        words.push_back(text.substr(start, i - start));
    }
    return words;
}

[[maybe_unused]] std::string metaspace_pre_tokenize(const std::string& text,
                                                    bool add_prefix_space) {
    std::string result;
    result.reserve(text.size() + 8);
    if (add_prefix_space && !text.empty() && text[0] != ' ') {
        result += kMetaspaceChar;
    }
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == ' ') {
            result += kMetaspaceChar;
        } else {
            result += text[i];
        }
    }
    return result;
}

// ─── Trie for efficient vocab prefix lookup ───

struct TrieNode {
    std::unordered_map<char, int> children;
    int token_id = -1; // -1 = not a token end
    float score = 0.0f;
};

class Trie {
  public:
    Trie() { mNodes.emplace_back(); } // root node

    void insert(const std::string& token, int id, float score) {
        int node = 0;
        for (char c : token) {
            auto it = mNodes[node].children.find(c);
            if (it == mNodes[node].children.end()) {
                int next = static_cast<int>(mNodes.size());
                mNodes.emplace_back();
                mNodes[node].children[c] = next;
                node = next;
            } else {
                node = it->second;
            }
        }
        mNodes[node].token_id = id;
        mNodes[node].score = score;
    }

    // Find all tokens that match a prefix of text starting at offset.
    // Returns vector of (token_id, byte_length, score).
    struct Match {
        int token_id;
        size_t length;
        float score;
    };

    void find_prefixes(const std::string& text, size_t offset, std::vector<Match>& out) const {
        out.clear();
        int node = 0;
        for (size_t i = offset; i < text.size(); ++i) {
            char c = text[i];
            auto it = mNodes[node].children.find(c);
            if (it == mNodes[node].children.end())
                break;
            node = it->second;
            if (mNodes[node].token_id >= 0) {
                out.push_back({mNodes[node].token_id, i - offset + 1, mNodes[node].score});
            }
        }
    }

  private:
    std::vector<TrieNode> mNodes;
};

// ─── Viterbi algorithm for Unigram tokenization ───

struct ViterbiNode {
    float score;
    int token_id;
    size_t prev_pos; // byte position of previous node
};

std::vector<int32_t> viterbi_encode(const std::string& text, const Trie& trie, int32_t unk_id,
                                    float unk_score) {
    if (text.empty())
        return {};

    size_t n = text.size();
    // best[i] = best path score to reach byte position i
    std::vector<ViterbiNode> best(n + 1, {-std::numeric_limits<float>::infinity(), -1, 0});
    best[0].score = 0.0f;

    std::vector<Trie::Match> matches;

    for (size_t i = 0; i < n; ++i) {
        if (best[i].score == -std::numeric_limits<float>::infinity())
            continue;

        trie.find_prefixes(text, i, matches);

        for (const auto& m : matches) {
            size_t end = i + m.length;
            float new_score = best[i].score + m.score;
            if (new_score > best[end].score) {
                best[end].score = new_score;
                best[end].token_id = m.token_id;
                best[end].prev_pos = i;
            }
        }

        // Advance by one UTF-8 codepoint after emitting UNK.
        size_t char_len = utf8_char_len(text, i);
        size_t end = i + char_len;
        if (end <= n) {
            float new_score = best[i].score + unk_score;
            if (new_score > best[end].score) {
                best[end].score = new_score;
                best[end].token_id = unk_id;
                best[end].prev_pos = i;
            }
        }
    }

    // Backtrack to find the best path
    std::vector<int32_t> ids;
    size_t pos = n;
    while (pos > 0) {
        if (best[pos].token_id < 0) {
            // Unreachable after emitting UNK.
            ids.push_back(unk_id);
            break;
        }
        ids.push_back(best[pos].token_id);
        pos = best[pos].prev_pos;
    }

    std::reverse(ids.begin(), ids.end());
    return ids;
}

// ─── UnigramTokenizer ───

class UnigramTokenizer final : public ITokenizer {
  public:
    static std::unique_ptr<UnigramTokenizer> Create(const char* json_data, std::size_t json_size,
                                                    bool add_special_tokens) {
        auto tok = std::unique_ptr<UnigramTokenizer>(new UnigramTokenizer());
        tok->mAddSpecialTokens = add_special_tokens;
        tok->parse_tokenizer_json(json_data, json_size);
        return tok;
    }

    std::vector<int32_t> encode(const std::string& text) const override {
        if (text.empty()) {
            return mAddSpecialTokens ? make_special_frame({}) : std::vector<int32_t>{};
        }

        // Normalize
        std::string normalized = mPrecompiled ? mPrecompiled->normalize(text) : text;
        if (mLowercase)
            normalized = lowercase_ascii(std::move(normalized));

        // Pre-tokenize: WhitespaceSplit → Metaspace
        auto words = whitespace_split(normalized);
        std::vector<int32_t> ids;
        for (size_t i = 0; i < words.size(); ++i) {
            // Metaspace: prepend ▁ to each word
            std::string processed = kMetaspaceChar + words[i];
            auto word_ids = viterbi_encode(processed, mTrie, mUnkId, mUnkScore);
            ids.insert(ids.end(), word_ids.begin(), word_ids.end());
        }

        if (mAddSpecialTokens)
            ids = make_special_frame(ids);
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string result;
        for (int32_t id : ids) {
            if (mDecodeSkipIds.count(id))
                continue;
            std::string token = token_for_id(id);
            result += token;
        }
        // Remove metaspace characters and clean up
        return decode_metaspace(result);
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
    UnigramTokenizer() = default;

    static std::string decode_metaspace(const std::string& text) {
        std::string result;
        size_t pos = 0;
        while (pos < text.size()) {
            if (pos + kMetaspaceChar.size() <= text.size() &&
                text.compare(pos, kMetaspaceChar.size(), kMetaspaceChar) == 0) {
                if (!result.empty())
                    result += ' ';
                pos += kMetaspaceChar.size();
            } else {
                result += text[pos];
                ++pos;
            }
        }
        return result;
    }

    std::vector<int32_t> make_special_frame(std::vector<int32_t> ids) const {
        std::vector<int32_t> result;
        if (mBosId >= 0)
            result.push_back(mBosId);
        result.insert(result.end(), ids.begin(), ids.end());
        if (mEosId >= 0)
            result.push_back(mEosId);
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
        parse_vocab(j);
        build_trie();
        parse_normalizer(j);
        parse_pre_tokenizer(j);
        parse_added_tokens(j);
        parse_post_processor(j);
        resolve_special_ids();
    }

    static void validate_model(const nlohmann::json& j) {
        if (!j.contains("model"))
            throw std::runtime_error("Invalid tokenizer.json: missing model");

        auto& model = j["model"];

        if (!model.contains("vocab") || !model["vocab"].is_array())
            throw std::runtime_error("Invalid tokenizer.json: model.vocab must be an array");
    }

    void parse_vocab(const nlohmann::json& j) {
        auto& vocab = j["model"]["vocab"];
        mIdToToken.resize(vocab.size());
        mUnkId = j["model"].value("unk_id", 0);

        for (size_t i = 0; i < vocab.size(); ++i) {
            auto& entry = vocab[i];
            std::string token = entry[0].get<std::string>();
            float score = entry[1].get<float>();

            mIdToToken[i] = token;
            mTokenToId[token] = static_cast<int32_t>(i);
            mScores.push_back(score);
        }

        // UNK score: must be worse than ANY real vocab token for Viterbi
        float min_score = 0.0f;
        for (float s : mScores) {
            if (s < min_score)
                min_score = s;
        }
        mUnkScore = min_score - 10.0f;
    }

    void build_trie() {
        for (size_t i = 0; i < mIdToToken.size(); ++i) {
            const auto& token = mIdToToken[i];
            if (!token.empty()) {
                mTrie.insert(token, static_cast<int>(i), mScores[i]);
            }
        }
    }

    void apply_normalizer_config(const nlohmann::json& norm) {
        const std::string ntype = norm.value("type", "");
        if (ntype == "Precompiled") {
            if (!norm.contains("precompiled_charsmap") ||
                !norm.at("precompiled_charsmap").is_string())
                throw std::runtime_error("Precompiled normalizer requires precompiled_charsmap");
            mPrecompiled = std::make_unique<PrecompiledNormalizer>(norm.at("precompiled_charsmap"));
            return;
        }
        if (ntype == "Lowercase") {
            mLowercase = true;
            return;
        }
        if (ntype == "Prepend") {
            // Marian-style: prepend a string (e.g., "▁") to input
            mAddPrefixSpace = true;
        }
    }

    void parse_normalizer(const nlohmann::json& j) {
        if (!j.contains("normalizer") || j["normalizer"].is_null())
            return;
        const auto& norm = j["normalizer"];
        if (norm.value("type", "") == "Sequence" && norm.contains("normalizers")) {
            for (const auto& sub : norm["normalizers"])
                apply_normalizer_config(sub);
            return;
        }
        apply_normalizer_config(norm);
    }

    void parse_pre_tokenizer(const nlohmann::json& j) {
        if (!j.contains("pre_tokenizer") || j["pre_tokenizer"].is_null())
            return;
        auto& pt = j["pre_tokenizer"];
        std::string ptype = pt.value("type", "");

        if (ptype == "Metaspace") {
            mAddPrefixSpace = pt.value("add_prefix_space", true);
        } else if (ptype == "Sequence") {
            // Look for Metaspace inside the sequence
            if (pt.contains("pretokenizers")) {
                for (auto& sub : pt["pretokenizers"]) {
                    if (sub.value("type", "") == "Metaspace") {
                        mAddPrefixSpace = sub.value("add_prefix_space", true);
                        break;
                    }
                }
            }
        }
    }

    void parse_added_tokens(const nlohmann::json& j) {
        if (!j.contains("added_tokens"))
            return;
        for (auto& tok : j["added_tokens"]) {
            int32_t tok_id = tok.value("id", -1);
            std::string content = tok.value("content", "");
            if (tok_id >= 0 && !content.empty()) {
                if (static_cast<size_t>(tok_id) >= mIdToToken.size()) {
                    mIdToToken.resize(static_cast<size_t>(tok_id) + 1);
                    mScores.resize(static_cast<size_t>(tok_id) + 1, 0.0f);
                }
                mIdToToken[tok_id] = content;
                mTokenToId[content] = tok_id;
            }
        }
    }

    // Extract BOS/EOS from TemplateProcessing "single" array
    void extract_template_bos_eos(const nlohmann::json& pp) {
        if (!pp.contains("single") || !pp["single"].is_array())
            return;
        bool seen_sequence = false;
        for (auto& item : pp["single"]) {
            if (item.contains("Sequence")) {
                seen_sequence = true;
                continue;
            }
            if (!item.contains("SpecialToken"))
                continue;
            std::string tok_str = item["SpecialToken"].value("id", "");
            auto it = mTokenToId.find(tok_str);
            if (it == mTokenToId.end())
                continue;
            if (!seen_sequence && mBosId < 0)
                mBosId = it->second;
            else
                mEosId = it->second;
        }
    }

    // Extract BOS/EOS from RobertaProcessing cls/sep arrays
    static int32_t extract_pp_id(const nlohmann::json& pp, const char* key) {
        if (pp.contains(key) && pp[key].is_array() && pp[key].size() >= 2)
            return pp[key][1].get<int32_t>();
        return -1;
    }

    void parse_post_processor(const nlohmann::json& j) {
        if (!j.contains("post_processor") || j["post_processor"].is_null())
            return;
        auto& pp = j["post_processor"];
        std::string ptype = pp.value("type", "");

        if (ptype == "TemplateProcessing")
            extract_template_bos_eos(pp);
        if (ptype == "RobertaProcessing") {
            mBosId = extract_pp_id(pp, "cls");
            mEosId = extract_pp_id(pp, "sep");
        }
    }

    void resolve_special_ids() {
        // Try common special-token names.
        auto find_id = [this](const std::string& a, const std::string& b) -> int32_t {
            auto it = mTokenToId.find(a);
            if (it != mTokenToId.end())
                return it->second;
            it = mTokenToId.find(b);
            return it != mTokenToId.end() ? it->second : -1;
        };

        if (mBosId < 0)
            mBosId = find_id("<s>", "[CLS]");
        if (mEosId < 0)
            mEosId = find_id("</s>", "[SEP]");
        int32_t padId = find_id("<pad>", "[PAD]");

        // Build decode skip set
        for (int32_t id : {mBosId, mEosId, padId}) {
            if (id >= 0)
                mDecodeSkipIds.insert(id);
        }
    }

    // ─── Data members ───

    std::vector<std::string> mIdToToken;
    std::vector<float> mScores;
    std::unordered_map<std::string, int32_t> mTokenToId;
    std::unordered_set<int32_t> mDecodeSkipIds;
    Trie mTrie;

    int32_t mUnkId = 0;
    float mUnkScore = -100.0f;
    bool mAddSpecialTokens = true;
    std::unique_ptr<PrecompiledNormalizer> mPrecompiled;
    bool mLowercase = false;
    bool mAddPrefixSpace = true;

    int32_t mBosId = -1;
    int32_t mEosId = -1;
};

} // namespace

std::unique_ptr<ITokenizer> CreateUnigramTokenizer(const char* tokenizer_json_data,
                                                   std::size_t tokenizer_json_size,
                                                   bool add_special_tokens) {
    return UnigramTokenizer::Create(tokenizer_json_data, tokenizer_json_size, add_special_tokens);
}

} // namespace trtmc
