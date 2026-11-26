/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esmc/runtime/tokenizer.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace trtmc {
namespace {

// Bytes in the UTF-8 sequence that starts with `lead`, or 1 when the byte does
// not start a well-formed sequence. An invalid byte then costs one <unk>, which
// is the smallest damage a caller can observe.
std::size_t utf8_width(unsigned char lead) {
    if (lead < 0x80)
        return 1;
    if ((lead & 0xE0) == 0xC0)
        return 2;
    if ((lead & 0xF0) == 0xE0)
        return 3;
    if ((lead & 0xF8) == 0xF0)
        return 4;
    return 1;
}

class EsmcTokenizer final : public IEsmcTokenizer {
  public:
    explicit EsmcTokenizer(std::vector<std::string> tokens) : tokens_(std::move(tokens)) {
        if (tokens_.empty())
            throw std::runtime_error("esmc vocab.txt is empty");
        for (std::size_t index = 0; index < tokens_.size(); ++index) {
            if (tokens_[index].empty())
                throw std::runtime_error("esmc vocab.txt has an empty token");
            // A duplicate would make the id for a token ambiguous. The
            // reference keeps the first, so this does too.
            ids_.emplace(tokens_[index], static_cast<std::int32_t>(index));
            longest_ = std::max(longest_, tokens_[index].size());
        }
        const auto unknown = ids_.find("<unk>");
        if (unknown == ids_.end())
            throw std::runtime_error("esmc vocab.txt does not define <unk>");
        unknown_ = unknown->second;
    }

    std::vector<std::int32_t> encode(const std::string& text) const override {
        std::vector<std::int32_t> ids;
        std::size_t position = 0;
        while (position < text.size()) {
            const auto matched = match(text, position);
            if (matched.second > 0) {
                ids.push_back(matched.first);
                position += matched.second;
                continue;
            }
            // One <unk> per character, not per unknown run: the reference maps
            // each character through an alphabet that does not carry it.
            const auto width = utf8_width(static_cast<unsigned char>(text[position]));
            position += std::min(width, text.size() - position);
            ids.push_back(unknown_);
        }
        return ids;
    }

    std::int32_t id_for_token(std::string_view token) const override {
        const auto found = ids_.find(std::string(token));
        return found == ids_.end() ? unknown_ : found->second;
    }

    std::string token_for_id(std::int32_t id) const override {
        if (id < 0 || static_cast<std::size_t>(id) >= tokens_.size())
            return tokens_[static_cast<std::size_t>(unknown_)];
        return tokens_[static_cast<std::size_t>(id)];
    }

    std::size_t size() const override { return tokens_.size(); }

  private:
    // Longest match at `position`, or {unknown, 0} when nothing matches.
    std::pair<std::int32_t, std::size_t> match(const std::string& text,
                                               std::size_t position) const {
        const auto available = text.size() - position;
        for (std::size_t length = std::min(longest_, available); length > 0; --length) {
            const auto found = ids_.find(text.substr(position, length));
            if (found != ids_.end())
                return {found->second, length};
        }
        return {unknown_, 0};
    }

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, std::int32_t> ids_;
    std::size_t longest_{0};
    std::int32_t unknown_{0};
};

} // namespace

std::unique_ptr<IEsmcTokenizer> CreateEsmcTokenizer(const char* vocab_data,
                                                    std::size_t vocab_size) {
    if (vocab_data == nullptr || vocab_size == 0)
        return nullptr;
    std::vector<std::string> tokens;
    std::string line;
    for (std::size_t index = 0; index < vocab_size; ++index) {
        const char value = vocab_data[index];
        if (value == '\n') {
            while (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (!line.empty())
                tokens.push_back(line);
            line.clear();
            continue;
        }
        line.push_back(value);
    }
    while (!line.empty() && line.back() == '\r')
        line.pop_back();
    if (!line.empty())
        tokens.push_back(line);
    if (tokens.empty())
        return nullptr;
    return std::make_unique<EsmcTokenizer>(std::move(tokens));
}

} // namespace trtmc
