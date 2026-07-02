/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace trtmc {
namespace {

class VocabTokenizer final : public ITokenizer {
public:
    explicit VocabTokenizer(std::vector<std::string> vocab)
    {
        mVocab = std::move(vocab);
        for (std::size_t i = 0; i < mVocab.size(); ++i)
        {
            mTokenToId.emplace(normalize(mVocab[i]), static_cast<int32_t>(i));
        }

        const auto it = mTokenToId.find("<unk>");
        mUnkId = (it == mTokenToId.end()) ? 0 : it->second;
        mBosId = id_for_token("<bos>");
        mEosId = id_for_token("<eos>");
        mPadId = id_for_token("<pad>");
    }

    std::vector<int32_t> encode(const std::string& text) const override
    {
        std::vector<int32_t> ids;
        ids.reserve(text.size() / 3 + 1);

        for (const auto& token : split_tokens(text))
        {
            ids.push_back(lookup_or_unk_id(token));
        }
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids) const override
    {
        std::ostringstream oss;
        bool first = true;
        for (const int32_t id : ids)
        {
            if (is_special_id(id))
            {
                continue;
            }
            append_decoded_token(oss, token_for_id(id), first);
        }
        return oss.str();
    }

    int32_t id_for_token(std::string_view token) const override
    {
        const auto key = normalize(token);
        const auto it = mTokenToId.find(key);
        if (it == mTokenToId.end())
        {
            return mUnkId;
        }
        return it->second;
    }

    std::string token_for_id(int32_t id) const override
    {
        if (id < 0 || static_cast<std::size_t>(id) >= mVocab.size())
        {
            return mVocab[static_cast<std::size_t>(mUnkId)];
        }
        return mVocab[static_cast<std::size_t>(id)];
    }

private:
    static std::string normalize(std::string_view token)
    {
        std::string out(token);
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return out;
    }

    static bool is_word_piece_char(unsigned char ch)
    {
        return std::isalnum(ch) != 0 || ch == '\'';
    }

    static bool is_punctuation_char(unsigned char ch)
    {
        return ch == '.' || ch == ',' || ch == '?' || ch == '!';
    }

    static void flush_current_token(std::string& current, std::vector<std::string>& tokens)
    {
        if (current.empty())
        {
            return;
        }

        tokens.push_back(normalize(current));
        current.clear();
    }

    int32_t lookup_or_unk_id(const std::string& token) const
    {
        const auto it = mTokenToId.find(token);
        if (it == mTokenToId.end())
        {
            return mUnkId;
        }
        return it->second;
    }

    bool is_special_id(int32_t id) const
    {
        return id == mBosId || id == mEosId || id == mPadId;
    }

    static bool is_punctuation_token(std::string_view token)
    {
        return token.size() == 1
            && is_punctuation_char(static_cast<unsigned char>(token.front()));
    }

    static void append_decoded_token(std::ostringstream& oss, const std::string& token, bool& first)
    {
        if (!first && !is_punctuation_token(token))
        {
            oss << ' ';
        }
        oss << token;
        first = false;
    }

    static std::vector<std::string> split_tokens(const std::string& text)
    {
        std::vector<std::string> tokens;
        std::string current;

        for (const unsigned char ch : text)
        {
            if (is_word_piece_char(ch))
            {
                current.push_back(static_cast<char>(ch));
                continue;
            }

            flush_current_token(current, tokens);
            if (is_punctuation_char(ch))
            {
                tokens.emplace_back(1, static_cast<char>(ch));
            }
        }

        flush_current_token(current, tokens);
        return tokens;
    }

    std::vector<std::string> mVocab;
    std::unordered_map<std::string, int32_t> mTokenToId;
    int32_t mUnkId{0};
    int32_t mBosId{0};
    int32_t mEosId{0};
    int32_t mPadId{0};
};

} // namespace

std::unique_ptr<ITokenizer> CreateVocabTokenizer(std::vector<std::string> vocab)
{
    if (vocab.empty())
    {
        throw std::invalid_argument("Vocabulary must not be empty.");
    }
    return std::make_unique<VocabTokenizer>(std::move(vocab));
}

} // namespace trtmc
