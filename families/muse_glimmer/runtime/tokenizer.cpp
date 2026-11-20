/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/muse_glimmer/runtime/tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <ctime>
#include <iomanip>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <unicode/regex.h>
#include <unicode/unistr.h>
#include <unordered_map>

namespace trtmc::muse_glimmer {
namespace {
using Pair = std::pair<std::string, std::string>;
struct PairHash {
    std::size_t operator()(const Pair& pair) const {
        return std::hash<std::string>{}(pair.first) ^ (std::hash<std::string>{}(pair.second) << 1);
    }
};
std::string bytes_from_token(const std::string& token) {
    static const auto inverse = [] {
        std::unordered_map<UChar32, unsigned char> result;
        int extension = 256;
        for (int byte = 0; byte < 256; ++byte) {
            const bool direct =
                (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
            result.emplace(direct ? byte : extension++, static_cast<unsigned char>(byte));
        }
        return result;
    }();
    const auto unicode = icu::UnicodeString::fromUTF8(token);
    std::string result;
    for (int32_t i = 0; i < unicode.length();) {
        const UChar32 ch = unicode.char32At(i);
        i += U16_LENGTH(ch);
        const auto found = inverse.find(ch);
        if (found == inverse.end())
            throw std::invalid_argument("Muse BPE vocabulary is not byte-level encoded");
        result.push_back(static_cast<char>(found->second));
    }
    return result;
}
std::string utc_date() {
    static std::mutex mutex;
    const std::lock_guard<std::mutex> guard(mutex);
    const auto now = std::time(nullptr);
    const auto* calendar = std::gmtime(&now);
    if (calendar == nullptr)
        throw std::runtime_error("Cannot determine UTC date for Muse ATEM template");
    std::ostringstream text;
    text << std::put_time(calendar, "%Y-%m-%d");
    return text.str();
}
void replace_all(std::string& text, const std::string& from, const std::string& to) {
    std::size_t position = 0;
    while ((position = text.find(from, position)) != std::string::npos) {
        text.replace(position, from.size(), to);
        position += to.size();
    }
}
} // namespace

struct Tokenizer::Impl {
    std::unordered_map<std::string, std::int32_t> vocab;
    std::unordered_map<Pair, std::size_t, PairHash> merges;
    std::vector<std::string> decoded;
    std::vector<std::pair<std::string, std::int32_t>> added;
    std::unique_ptr<icu::RegexPattern> pattern;

    void append_piece(const std::string& text, std::vector<std::int32_t>& output) const {
        std::vector<std::string> pieces;
        for (const char ch : text)
            pieces.emplace_back(1, ch);
        while (pieces.size() > 1) {
            auto best = std::numeric_limits<std::size_t>::max();
            std::size_t at = 0;
            for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
                const auto found = merges.find({pieces[i], pieces[i + 1]});
                if (found != merges.end() && found->second < best) {
                    best = found->second;
                    at = i;
                }
            }
            if (best == std::numeric_limits<std::size_t>::max())
                break;
            pieces[at] += pieces[at + 1];
            pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(at + 1));
        }
        for (const auto& piece : pieces) {
            const auto found = vocab.find(piece);
            if (found == vocab.end())
                throw std::runtime_error("Muse byte BPE produced an unknown token");
            output.push_back(found->second);
        }
    }

    void append_text(const std::string& text, std::vector<std::int32_t>& output) const {
        const auto input = icu::UnicodeString::fromUTF8(text);
        UErrorCode status = U_ZERO_ERROR;
        std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(input, status));
        if (U_FAILURE(status))
            throw std::runtime_error("Muse regex matcher creation failed");
        int32_t previous = 0;
        while (matcher->find(status)) {
            const auto begin = matcher->start(status);
            const auto end = matcher->end(status);
            if (begin > previous) {
                std::string gap;
                input.tempSubStringBetween(previous, begin).toUTF8String(gap);
                append_piece(gap, output);
            }
            if (end <= begin)
                throw std::runtime_error("Muse tokenizer regex matched an empty string");
            std::string piece;
            input.tempSubStringBetween(begin, end).toUTF8String(piece);
            append_piece(piece, output);
            previous = end;
        }
        if (U_FAILURE(status))
            throw std::runtime_error("Muse tokenizer regex matching failed");
        if (previous < input.length()) {
            std::string rest;
            input.tempSubString(previous).toUTF8String(rest);
            append_piece(rest, output);
        }
    }
};

Tokenizer::Tokenizer(const std::string& text) : impl_(std::make_unique<Impl>()) {
    const auto json = nlohmann::json::parse(text);
    const auto& model = json.at("model");
    if (model.at("type") != "BPE" || !json.at("normalizer").is_null() ||
        json.at("decoder").at("type") != "ByteLevel")
        throw std::invalid_argument("Muse requires its unnormalized byte-level BPE tokenizer");
    const auto& pre = json.at("pre_tokenizer");
    const auto& stages = pre.at("pretokenizers");
    if (pre.at("type") != "Sequence" || stages.size() != 2 || stages[0].at("type") != "Split" ||
        stages[0].at("behavior") != "Isolated" || stages[0].value("invert", false) ||
        stages[1].at("type") != "ByteLevel" || stages[1].value("add_prefix_space", false) ||
        stages[1].value("use_regex", true))
        throw std::invalid_argument(
            "Muse tokenizer pretokenization contract differs from the source recipe");
    UErrorCode status = U_ZERO_ERROR;
    impl_->pattern.reset(icu::RegexPattern::compile(
        icu::UnicodeString::fromUTF8(stages[0].at("pattern").at("Regex").get<std::string>()), 0,
        status));
    if (U_FAILURE(status))
        throw std::invalid_argument("Muse tokenizer regex is not supported by ICU");
    for (const auto& entry : model.at("vocab").items()) {
        const auto id = entry.value().get<std::int32_t>();
        if (id < 0)
            throw std::invalid_argument("Muse vocabulary has a negative ID");
        const auto bytes = bytes_from_token(entry.key());
        if (!impl_->vocab.emplace(bytes, id).second)
            throw std::invalid_argument("Muse vocabulary contains duplicate byte strings");
        if (impl_->decoded.size() <= static_cast<std::size_t>(id))
            impl_->decoded.resize(static_cast<std::size_t>(id) + 1);
        impl_->decoded[id] = bytes;
    }
    std::size_t rank = 0;
    for (const auto& merge : model.at("merges")) {
        if (!merge.is_array() || merge.size() != 2)
            throw std::invalid_argument("Muse requires explicit pair-valued BPE merges");
        impl_->merges.emplace(Pair{bytes_from_token(merge[0]), bytes_from_token(merge[1])}, rank++);
    }
    for (const auto& item : json.at("added_tokens")) {
        if (item.value("single_word", false) || item.value("lstrip", false) ||
            item.value("rstrip", false) || item.value("normalized", false))
            throw std::invalid_argument("Muse added-token matching policy is unsupported");
        const auto content = item.at("content").get<std::string>();
        const auto id = item.at("id").get<std::int32_t>();
        if (content.empty() || id < 0)
            throw std::invalid_argument("Muse added token has invalid content or ID");
        impl_->added.emplace_back(content, id);
        if (impl_->decoded.size() <= static_cast<std::size_t>(id))
            impl_->decoded.resize(static_cast<std::size_t>(id) + 1);
        impl_->decoded[id] = content;
    }
}
Tokenizer::~Tokenizer() = default;

std::vector<std::int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<std::int32_t> output;
    std::size_t position = 0;
    while (position < text.size()) {
        std::size_t earliest = text.size();
        const std::pair<std::string, std::int32_t>* selected = nullptr;
        for (const auto& token : impl_->added) {
            const auto found = text.find(token.first, position);
            if (found < earliest || (found == earliest && selected != nullptr &&
                                     token.first.size() > selected->first.size())) {
                earliest = found;
                selected = &token;
            }
        }
        impl_->append_text(text.substr(position, earliest - position), output);
        if (selected == nullptr)
            break;
        output.push_back(selected->second);
        position = earliest + selected->first.size();
    }
    return output;
}
std::string Tokenizer::decode(const std::vector<std::int32_t>& ids) const {
    std::string result;
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= impl_->decoded.size())
            throw std::out_of_range("Muse token ID is outside the vocabulary");
        result += impl_->decoded[id];
    }
    return result;
}
std::string render_atem(const std::string& prompt, const std::string& system,
                        const std::string& bos) {
    auto text = system;
    if (text.empty())
        text = "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent date: " +
               utc_date() + ".";
    for (const auto& pair : std::array<Pair, 4>{{{"Reasoning effort", "Reasoning strength"},
                                                 {"Reasoning Effort", "Reasoning Strength"},
                                                 {"reasoning effort", "reasoning strength"},
                                                 {"REASONING EFFORT", "REASONING STRENGTH"}}})
        replace_all(text, pair.first, pair.second);
    auto lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (lower.find("reasoning strength") == std::string::npos)
        text += "\n\nReasoning strength: high.";
    return bos + "<|start|>system<|message|>" + text +
           "\n\n# Valid recipients: \"self\", \"user\".<|eot|>" + "<|start|>user<|message|>" +
           prompt + "<|eot|><|start|>assistant";
}
} // namespace trtmc::muse_glimmer
