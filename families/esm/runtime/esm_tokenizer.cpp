/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esm/runtime/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <unordered_map>

namespace trtmc {
namespace {

bool is_space(unsigned char value) {
    return std::isspace(value) != 0;
}

class EsmTokenizer final : public IEsmTokenizer {
  public:
    explicit EsmTokenizer(std::vector<std::string> tokens) : tokens_(std::move(tokens)) {
        if (tokens_.empty())
            throw std::runtime_error("esm vocab.txt is empty");
        for (std::size_t index = 0; index < tokens_.size(); ++index) {
            if (tokens_[index].empty())
                throw std::runtime_error("esm vocab.txt has an empty token");
            // A duplicate would make the id for a token ambiguous. The
            // reference keeps the first, so this does too.
            ids_.emplace(tokens_[index], static_cast<std::int32_t>(index));
            longest_ = std::max(longest_, tokens_[index].size());
        }
        const auto unknown = ids_.find("<unk>");
        if (unknown == ids_.end())
            throw std::runtime_error("esm vocab.txt does not define <unk>");
        unknown_ = unknown->second;
    }

    std::vector<std::int32_t> encode(const std::string& text) const override {
        std::vector<std::int32_t> ids;
        std::size_t position = 0;
        while (position < text.size()) {
            if (is_space(static_cast<unsigned char>(text[position]))) {
                ++position;
                continue;
            }
            const auto matched = match(text, position);
            if (matched.second > 0) {
                ids.push_back(matched.first);
                position += matched.second;
                continue;
            }
            // Nothing in the vocabulary starts here. The reference hands the
            // leftover run to split() and maps each piece through the vocab,
            // where it misses and becomes <unk>; one <unk> per run is the
            // same result.
            std::size_t end = position;
            while (end < text.size() && !is_space(static_cast<unsigned char>(text[end])) &&
                   match(text, end).second == 0) {
                ++end;
            }
            ids.push_back(unknown_);
            position = end;
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

std::unique_ptr<IEsmTokenizer> CreateEsmTokenizer(const char* vocab_data, std::size_t vocab_size) {
    if (vocab_data == nullptr || vocab_size == 0)
        return nullptr;
    std::vector<std::string> tokens;
    std::string line;
    for (std::size_t index = 0; index < vocab_size; ++index) {
        const char value = vocab_data[index];
        if (value == '\n') {
            while (!line.empty() &&
                   (line.back() == '\r' || is_space(static_cast<unsigned char>(line.back())))) {
                line.pop_back();
            }
            if (!line.empty())
                tokens.push_back(line);
            line.clear();
            continue;
        }
        line.push_back(value);
    }
    while (!line.empty() && is_space(static_cast<unsigned char>(line.back())))
        line.pop_back();
    if (!line.empty())
        tokens.push_back(line);
    if (tokens.empty())
        return nullptr;
    return std::make_unique<EsmTokenizer>(std::move(tokens));
}

} // namespace trtmc
