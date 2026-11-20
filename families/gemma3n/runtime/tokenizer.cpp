/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/gemma3n/runtime/tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <nlohmann/json.hpp>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace trtmc::gemma3n {
namespace {

// U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's visible space.
constexpr std::string_view kSpaceMark{"\xE2\x96\x81"};

std::size_t utf8_length(unsigned char lead) {
    if (lead < 0x80)
        return 1;
    if ((lead >> 5) == 0x6)
        return 2;
    if ((lead >> 4) == 0xE)
        return 3;
    if ((lead >> 3) == 0x1E)
        return 4;
    throw std::invalid_argument("Gemma 3n tokenizer input is not valid UTF-8");
}

struct Symbol {
    int32_t id;
    int32_t prev;
    int32_t next;
    bool removed;
};

struct Merge {
    uint32_t rank;
    int32_t position;
    int32_t new_id;
};

// Lowest rank first, then the leftmost position (the reference's merge order).
struct LaterMerge {
    bool operator()(const Merge& a, const Merge& b) const {
        return a.rank != b.rank ? a.rank > b.rank : a.position > b.position;
    }
};

uint64_t pair_key(int32_t left, int32_t right) {
    return (uint64_t(uint32_t(left)) << 32) | uint32_t(right);
}

} // namespace

struct Tokenizer::Impl {
    std::unordered_map<std::string, int32_t> vocab;
    std::vector<std::string> tokens;
    std::unordered_map<uint64_t, std::pair<uint32_t, int32_t>> merges;
    std::unordered_map<std::string, int32_t> added;
    std::vector<bool> special;
    std::set<std::size_t, std::greater<>> added_lengths;
    std::array<bool, 256> added_first_byte{};
    std::array<int32_t, 256> byte_tokens{};

    void bpe(std::string_view segment, std::vector<int32_t>& output) const {
        std::string word;
        word.reserve(segment.size() + 8);
        for (const char c : segment) {
            if (c == ' ')
                word.append(kSpaceMark);
            else
                word.push_back(c);
        }
        std::vector<Symbol> symbols;
        for (std::size_t i = 0; i < word.size();) {
            const auto length = utf8_length(static_cast<unsigned char>(word[i]));
            if (i + length > word.size())
                throw std::invalid_argument("Gemma 3n tokenizer input is not valid UTF-8");
            const auto found = vocab.find(word.substr(i, length));
            if (found != vocab.end()) {
                symbols.push_back({found->second, 0, 0, false});
            } else {
                for (std::size_t b = 0; b < length; ++b)
                    symbols.push_back(
                        {byte_tokens[static_cast<unsigned char>(word[i + b])], 0, 0, false});
            }
            i += length;
        }
        const auto count = static_cast<int32_t>(symbols.size());
        for (int32_t i = 0; i < count; ++i) {
            symbols[size_t(i)].prev = i - 1;
            symbols[size_t(i)].next = i + 1 < count ? i + 1 : -1;
        }
        std::priority_queue<Merge, std::vector<Merge>, LaterMerge> queue;
        auto push = [&](int32_t position) {
            const auto next = symbols[size_t(position)].next;
            if (next < 0)
                return;
            const auto found =
                merges.find(pair_key(symbols[size_t(position)].id, symbols[size_t(next)].id));
            if (found != merges.end())
                queue.push({found->second.first, position, found->second.second});
        };
        for (int32_t i = 0; i + 1 < count; ++i)
            push(i);
        while (!queue.empty()) {
            const auto top = queue.top();
            queue.pop();
            auto& left = symbols[size_t(top.position)];
            if (left.removed || left.next < 0)
                continue;
            auto& right = symbols[size_t(left.next)];
            const auto current = merges.find(pair_key(left.id, right.id));
            // Entries whose pair has changed since they were queued are stale.
            if (current == merges.end() || current->second.second != top.new_id)
                continue;
            left.id = top.new_id;
            right.removed = true;
            left.next = right.next;
            if (right.next >= 0)
                symbols[size_t(right.next)].prev = top.position;
            if (left.prev >= 0)
                push(left.prev);
            push(top.position);
        }
        for (const auto& symbol : symbols)
            if (!symbol.removed)
                output.push_back(symbol.id);
    }
};

Tokenizer::Tokenizer(std::string_view serialized) : impl_(std::make_unique<Impl>()) {
    const auto document = nlohmann::json::parse(serialized);
    const auto& model = document.at("model");
    if (model.at("type") != "BPE" || model.at("byte_fallback") != true)
        throw std::invalid_argument("Gemma 3n requires a byte-fallback BPE tokenizer");
    auto& impl = *impl_;
    const auto& vocab = model.at("vocab");
    impl.tokens.resize(vocab.size());
    impl.vocab.reserve(vocab.size());
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const auto id = it.value().get<int32_t>();
        if (id < 0 || size_t(id) >= impl.tokens.size() || !impl.tokens[size_t(id)].empty())
            throw std::invalid_argument("Gemma 3n tokenizer vocabulary ids are not dense");
        impl.tokens[size_t(id)] = it.key();
        impl.vocab.emplace(it.key(), id);
    }
    for (int byte = 0; byte < 256; ++byte) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", byte);
        const auto found = impl.vocab.find(name);
        if (found == impl.vocab.end())
            throw std::invalid_argument("Gemma 3n tokenizer lacks a byte-fallback token");
        impl.byte_tokens[size_t(byte)] = found->second;
    }
    const auto& merges = model.at("merges");
    impl.merges.reserve(merges.size());
    uint32_t rank = 0;
    for (const auto& pair : merges) {
        const auto left = pair.at(0).get<std::string>();
        const auto right = pair.at(1).get<std::string>();
        const auto a = impl.vocab.find(left), b = impl.vocab.find(right),
                   merged = impl.vocab.find(left + right);
        if (a == impl.vocab.end() || b == impl.vocab.end() || merged == impl.vocab.end())
            throw std::invalid_argument("Gemma 3n tokenizer merge is outside the vocabulary");
        // A repeated pair keeps its last rank, as a map built in order does.
        impl.merges[pair_key(a->second, b->second)] = {rank++, merged->second};
    }
    impl.special.assign(impl.tokens.size(), false);
    for (const auto& token : document.at("added_tokens")) {
        const auto content = token.at("content").get<std::string>();
        const auto id = token.at("id").get<int32_t>();
        if (content.empty() || id < 0 || size_t(id) >= impl.tokens.size())
            throw std::invalid_argument("Invalid Gemma 3n added token");
        impl.added[content] = id;
        impl.added_lengths.insert(content.size());
        impl.added_first_byte[static_cast<unsigned char>(content[0])] = true;
        impl.special[size_t(id)] = token.at("special").get<bool>();
    }
}

Tokenizer::~Tokenizer() = default;

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    const auto& impl = *impl_;
    std::vector<int32_t> ids;
    std::size_t segment = 0;
    for (std::size_t i = 0; i < text.size();) {
        int32_t match = -1;
        std::size_t length = 0;
        if (impl.added_first_byte[static_cast<unsigned char>(text[i])]) {
            for (const auto candidate : impl.added_lengths) {
                if (candidate > text.size() - i)
                    continue;
                const auto found = impl.added.find(std::string(text.substr(i, candidate)));
                if (found != impl.added.end()) {
                    match = found->second;
                    length = candidate;
                    break;
                }
            }
        }
        if (match < 0) {
            ++i;
            continue;
        }
        if (segment < i)
            impl.bpe(text.substr(segment, i - segment), ids);
        ids.push_back(match);
        i += length;
        segment = i;
    }
    if (segment < text.size())
        impl.bpe(text.substr(segment), ids);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    const auto& impl = *impl_;
    std::string text, bytes;
    auto flush = [&] {
        if (bytes.empty())
            return;
        bool valid = true;
        for (std::size_t i = 0; i < bytes.size() && valid;) {
            const auto lead = static_cast<unsigned char>(bytes[i]);
            std::size_t length = 0;
            if (lead < 0x80)
                length = 1;
            else if ((lead >> 5) == 0x6 && lead >= 0xC2)
                length = 2;
            else if ((lead >> 4) == 0xE)
                length = 3;
            else if ((lead >> 3) == 0x1E && lead <= 0xF4)
                length = 4;
            if (length == 0 || i + length > bytes.size()) {
                valid = false;
                break;
            }
            uint32_t code = length == 1 ? lead : lead & (0x7F >> length);
            for (std::size_t k = 1; k < length; ++k) {
                const auto next = static_cast<unsigned char>(bytes[i + k]);
                if ((next >> 6) != 0x2)
                    valid = false;
                code = (code << 6) | (next & 0x3F);
            }
            // Reject overlong forms, surrogates and values beyond U+10FFFF.
            static constexpr uint32_t kMinimum[5] = {0, 0, 0x80, 0x800, 0x10000};
            if (code < kMinimum[length] || (code >= 0xD800 && code <= 0xDFFF) || code > 0x10FFFF)
                valid = false;
            i += length;
        }
        if (valid) {
            text += bytes;
        } else {
            // One replacement character per byte of an invalid sequence.
            for (std::size_t i = 0; i < bytes.size(); ++i)
                text += "\xEF\xBF\xBD";
        }
        bytes.clear();
    };
    for (const auto id : ids) {
        if (id < 0 || size_t(id) >= impl.tokens.size())
            throw std::invalid_argument("Gemma 3n token id is outside the vocabulary");
        if (impl.special[size_t(id)])
            continue;
        const auto& token = impl.tokens[size_t(id)];
        if (token.size() == 6 && token.compare(0, 3, "<0x") == 0 && token[5] == '>' &&
            std::isxdigit(static_cast<unsigned char>(token[3])) &&
            std::isxdigit(static_cast<unsigned char>(token[4]))) {
            bytes.push_back(
                static_cast<char>(std::strtol(token.substr(3, 2).c_str(), nullptr, 16)));
            continue;
        }
        flush();
        for (std::size_t i = 0; i < token.size();) {
            if (token.compare(i, kSpaceMark.size(), kSpaceMark) == 0) {
                text.push_back(' ');
                i += kSpaceMark.size();
            } else {
                text.push_back(token[i++]);
            }
        }
    }
    flush();
    return text;
}

int32_t Tokenizer::token_id(std::string_view token) const {
    const auto found = impl_->vocab.find(std::string(token));
    if (found == impl_->vocab.end())
        throw std::invalid_argument("Gemma 3n tokenizer lacks token " + std::string(token));
    return found->second;
}

int32_t Tokenizer::vocab_size() const {
    return static_cast<int32_t>(impl_->tokens.size());
}

} // namespace trtmc::gemma3n
