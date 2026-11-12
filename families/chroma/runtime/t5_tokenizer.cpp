/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/chroma/runtime/t5_tokenizer.h"

#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::chroma {
namespace {

using nlohmann::json;

constexpr std::string_view kMetaspace = "\xe2\x96\x81"; // U+2581
constexpr double kUnknownPenalty = 10.0;                // tokenizers' Unigram K_UNK_PENALTY

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(std::string("Chroma tokenizer: ") + message);
}

// Length of the UTF-8 sequence at text[position], or 0 when it is invalid.
std::size_t utf8_length(std::string_view text, std::size_t position, char32_t& code) {
    const auto lead = static_cast<unsigned char>(text[position]);
    std::size_t length = 0;
    char32_t value = 0;
    char32_t minimum = 0;
    if (lead < 0x80) {
        code = lead;
        return 1;
    }
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
    } else {
        return 0;
    }
    if (position + length > text.size())
        return 0;
    for (std::size_t index = 1; index < length; ++index) {
        const auto byte = static_cast<unsigned char>(text[position + index]);
        if ((byte & 0xC0) != 0x80)
            return 0;
        value = (value << 6) | (byte & 0x3F);
    }
    if (value < minimum || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF))
        return 0;
    code = value;
    return length;
}

// Unicode White_Space, the set Rust's char::is_whitespace (WhitespaceSplit) uses.
bool is_whitespace(char32_t code) {
    return (code >= 0x09 && code <= 0x0D) || code == 0x20 || code == 0x85 || code == 0xA0 ||
           code == 0x1680 || (code >= 0x2000 && code <= 0x200A) || code == 0x2028 ||
           code == 0x2029 || code == 0x202F || code == 0x205F || code == 0x3000;
}

void validate_configuration(const json& root) {
    require(root.contains("normalizer") && root.at("normalizer").is_null(),
            "a normalizer is not supported");
    const json expected_pre = {
        {"type", "Sequence"},
        {"pretokenizers", json::array({{{"type", "WhitespaceSplit"}},
                                       {{"type", "Metaspace"},
                                        {"replacement", std::string(kMetaspace)},
                                        {"prepend_scheme", "always"},
                                        {"split", true}}})}};
    require(root.at("pre_tokenizer") == expected_pre,
            "only WhitespaceSplit + always-prepended splitting Metaspace is supported");
    const auto& post = root.at("post_processor");
    const json single = json::array({{{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                                     {{"SpecialToken", {{"id", "</s>"}, {"type_id", 0}}}}});
    require(post.at("type") == "TemplateProcessing" && post.at("single") == single,
            "the post-processor must append </s> only");
    const auto& model = root.at("model");
    require(model.at("type") == "Unigram", "the model must be Unigram");
    require(!model.value("byte_fallback", false), "byte fallback is not supported");
}

} // namespace

T5Tokenizer::T5Tokenizer(const char* data, std::size_t size) {
    const auto root = json::parse(data, data + size);
    validate_configuration(root);
    const auto& model = root.at("model");
    unk_id_ = model.at("unk_id").get<std::int32_t>();
    const auto& vocab = model.at("vocab");
    require(vocab.is_array() && !vocab.empty(), "the vocabulary is empty");
    trie_.emplace_back();
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < vocab.size(); ++index) {
        const auto piece = vocab[index].at(0).get<std::string>();
        const double score = vocab[index].at(1).get<double>();
        require(!piece.empty(), "the vocabulary has an empty piece");
        const auto id = static_cast<std::int32_t>(index);
        require(ids_.emplace(piece, id).second, "the vocabulary has a duplicate piece");
        scores_.push_back(score);
        minimum = std::min(minimum, score);
        insert(piece, id);
    }
    require(unk_id_ >= 0 && static_cast<std::size_t>(unk_id_) < scores_.size(),
            "unk_id is out of range");
    unk_score_ = minimum - kUnknownPenalty;
    const auto eos = ids_.find("</s>");
    const auto pad = ids_.find("<pad>");
    require(eos != ids_.end() && pad != ids_.end(), "</s> and <pad> must be vocabulary pieces");
    eos_id_ = eos->second;
    pad_id_ = pad->second;
    for (const auto& token : root.value("added_tokens", json::array())) {
        require(!token.value("normalized", false) && !token.value("lstrip", false) &&
                    !token.value("rstrip", false) && !token.value("single_word", false),
                "added tokens with normalization or stripping are not supported");
        added_.push_back(
            {token.at("content").get<std::string>(), token.at("id").get<std::int32_t>()});
        require(!added_.back().content.empty(), "an added token is empty");
    }
}

void T5Tokenizer::insert(const std::string& piece, std::int32_t id) {
    std::int32_t node = 0;
    for (const char character : piece) {
        const auto byte = static_cast<unsigned char>(character);
        const auto found = trie_[static_cast<std::size_t>(node)].children.find(byte);
        if (found != trie_[static_cast<std::size_t>(node)].children.end()) {
            node = found->second;
            continue;
        }
        const auto next = static_cast<std::int32_t>(trie_.size());
        trie_[static_cast<std::size_t>(node)].children.emplace(byte, next);
        trie_.emplace_back();
        node = next;
    }
    trie_[static_cast<std::size_t>(node)].token = id;
}

// tokenizers' Unigram::encode_optimized followed by its fuse_unk backtrack.
void T5Tokenizer::encode_piece(std::string_view piece, std::vector<std::int32_t>& ids) const {
    struct Best {
        std::int32_t id{-1};
        double score{0.0};
        std::size_t start{0};
        bool set{false};
    };
    const std::size_t size = piece.size();
    std::vector<Best> best(size + 1);
    best[0].set = true;
    std::size_t start = 0;
    while (start < size) {
        char32_t code = 0;
        const std::size_t width = utf8_length(piece, start, code);
        const double base = best[start].score;
        bool single = false;
        std::int32_t node = 0;
        for (std::size_t end = start; end < size; ++end) {
            const auto& children = trie_[static_cast<std::size_t>(node)].children;
            const auto found = children.find(static_cast<unsigned char>(piece[end]));
            if (found == children.end())
                break;
            node = found->second;
            const std::int32_t token = trie_[static_cast<std::size_t>(node)].token;
            if (token < 0)
                continue;
            auto& target = best[end + 1];
            const double candidate = base + scores_[static_cast<std::size_t>(token)];
            if (!target.set || candidate > target.score)
                target = {token, candidate, start, true};
            if (end + 1 - start == width)
                single = true;
        }
        if (!single) {
            auto& target = best[start + width];
            const double candidate = base + unk_score_;
            if (!target.set || candidate > target.score)
                target = {unk_id_, candidate, start, true};
        }
        start += width;
    }
    std::vector<std::pair<std::size_t, std::size_t>> spans; // reversed [begin, end) token spans
    std::size_t end = size;
    std::size_t unknown_end = 0;
    bool in_unknown = false;
    while (end > 0) {
        const auto& node = best[end];
        if (node.id == unk_id_) {
            if (!in_unknown)
                unknown_end = end;
            in_unknown = true;
        } else {
            if (in_unknown)
                spans.emplace_back(end, unknown_end);
            in_unknown = false;
            spans.emplace_back(node.start, end);
        }
        end = node.start;
    }
    if (in_unknown)
        spans.emplace_back(0, unknown_end);
    for (auto span = spans.rbegin(); span != spans.rend(); ++span) {
        const auto found =
            ids_.find(std::string(piece.substr(span->first, span->second - span->first)));
        ids.push_back(found == ids_.end() ? unk_id_ : found->second);
    }
}

void T5Tokenizer::encode_text(std::string_view text, std::vector<std::int32_t>& ids) const {
    std::size_t position = 0;
    while (position < text.size()) {
        char32_t code = 0;
        std::size_t width = utf8_length(text, position, code);
        if (is_whitespace(code)) {
            position += width;
            continue;
        }
        std::size_t end = position;
        while (end < text.size()) {
            width = utf8_length(text, end, code);
            if (is_whitespace(code))
                break;
            end += width;
        }
        // Metaspace: prepend U+2581 unless present, then split before each U+2581.
        std::string word(text.substr(position, end - position));
        if (word.compare(0, kMetaspace.size(), kMetaspace) != 0)
            word.insert(0, kMetaspace);
        std::size_t begin = 0;
        while (begin < word.size()) {
            std::size_t next = word.find(kMetaspace, begin + kMetaspace.size());
            if (next == std::string::npos)
                next = word.size();
            encode_piece(std::string_view(word).substr(begin, next - begin), ids);
            begin = next;
        }
        position = end;
    }
}

std::vector<std::int32_t> T5Tokenizer::encode(std::string_view text) const {
    for (std::size_t position = 0; position < text.size();) {
        char32_t code = 0;
        const std::size_t width = utf8_length(text, position, code);
        require(width != 0, "the prompt is not valid UTF-8");
        position += width;
    }
    std::vector<std::int32_t> ids;
    std::size_t segment = 0;
    std::size_t position = 0;
    // Special tokens are matched leftmost-longest in the raw text first.
    while (position < text.size()) {
        const AddedToken* match = nullptr;
        for (const auto& token : added_) {
            if (text.compare(position, token.content.size(), token.content) == 0 &&
                (match == nullptr || token.content.size() > match->content.size()))
                match = &token;
        }
        if (match == nullptr) {
            ++position;
            continue;
        }
        encode_text(text.substr(segment, position - segment), ids);
        ids.push_back(match->id);
        position += match->content.size();
        segment = position;
    }
    encode_text(text.substr(segment), ids);
    ids.push_back(eos_id_);
    return ids;
}

TokenizedPrompt T5Tokenizer::encode_padded(std::string_view text, std::size_t length) const {
    require(length >= 1, "the text length must be positive");
    auto ids = encode(text);
    if (ids.size() > length) {
        ids.resize(length - 1);
        ids.push_back(eos_id_);
    }
    TokenizedPrompt result;
    result.valid = ids.size();
    result.ids = std::move(ids);
    result.mask.assign(result.valid, 1);
    result.ids.resize(length, pad_id_);
    result.mask.resize(length, 0);
    return result;
}

} // namespace trtmc::chroma
