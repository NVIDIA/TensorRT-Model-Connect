/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/phonon/runtime/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::phonon {
Tokenizer::Tokenizer(const std::string& json) {
    const auto doc = nlohmann::json::parse(json);
    if (doc.at("model").at("type") != "Unigram")
        throw std::invalid_argument("Phonon requires its Unigram tokenizer");
    for (const auto& entry : doc.at("model").at("vocab"))
        vocab_.emplace_back(entry.at(0).get<std::string>(), entry.at(1).get<double>());
    if (vocab_.size() != 4000)
        throw std::invalid_argument("Phonon tokenizer vocabulary must contain 4000 entries");
}

std::string Tokenizer::prepare(std::string_view text) {
    std::string result;
    bool space = false;
    for (const unsigned char c : text) {
        if (c >= 128 || (c < 32 && !std::isspace(c)))
            throw std::invalid_argument("Phonon currently accepts ASCII English text only");
        if (std::isspace(c)) {
            space = !result.empty();
            continue;
        }
        if (space) {
            result += ' ';
            space = false;
        }
        result += static_cast<char>(c);
    }
    if (result.empty())
        throw std::invalid_argument("Phonon requires nonempty text");
    // Fail explicitly on text requiring normalization beyond the initial path.
    if (result.find_first_of(";()[]{}*<>|\\") != std::string::npos)
        throw std::invalid_argument("Phonon requires normalized English prose");
    result[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(result[0])));
    if (std::isalnum(static_cast<unsigned char>(result.back())))
        result += '.';
    return result;
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text) const {
    const auto prepared = prepare(text);
    std::vector<std::int32_t> result;
    std::size_t first = 0;
    while (first < prepared.size()) {
        const auto end = prepared.find(' ', first);
        const auto word = std::string("\xe2\x96\x81") + prepared.substr(first, end - first);
        std::vector<double> score(word.size() + 1, -std::numeric_limits<double>::infinity());
        std::vector<std::size_t> previous(word.size() + 1);
        std::vector<std::int32_t> token(word.size() + 1, -1);
        score[0] = 0;
        for (std::size_t pos = 0; pos < word.size(); ++pos) {
            for (std::size_t id = 0; id < vocab_.size(); ++id) {
                const auto& [piece, value] = vocab_[id];
                if (piece.empty() || piece[0] == '\0' || piece.rfind("<0x", 0) == 0 ||
                    word.compare(pos, piece.size(), piece) != 0)
                    continue;
                const auto next = pos + piece.size();
                if (score[pos] + value > score[next]) {
                    score[next] = score[pos] + value;
                    previous[next] = pos;
                    token[next] = static_cast<std::int32_t>(id);
                }
            }
        }
        if (token.back() < 0)
            throw std::invalid_argument("Phonon text contains a character outside its vocabulary");
        std::vector<std::int32_t> ids;
        for (auto pos = word.size(); pos != 0; pos = previous[pos])
            ids.push_back(token[pos]);
        result.insert(result.end(), ids.rbegin(), ids.rend());
        if (end == std::string::npos)
            break;
        first = end + 1;
    }
    return result;
}
} // namespace trtmc::phonon
