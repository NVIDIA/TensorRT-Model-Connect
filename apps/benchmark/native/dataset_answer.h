/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <optional>
#include <regex>
#include <string>

namespace trtmc::examples::dataset_benchmark {
namespace detail {

inline std::optional<std::string> normalize_integer(std::string value) {
    value.erase(std::remove(value.begin(), value.end(), ','), value.end());
    static const std::regex integer("-?\\d+");
    std::smatch match;
    if (!std::regex_search(value, match, integer))
        return std::nullopt;
    return match.str(0);
}

inline void capture_last_phrase(const std::string& text, const std::regex& pattern,
                                std::optional<std::string>& answer) {
    for (std::sregex_iterator iterator(text.begin(), text.end(), pattern), end; iterator != end;
         ++iterator) {
        if (auto normalized = normalize_integer((*iterator).str(1)))
            answer = normalized;
    }
}

} // namespace detail

inline std::optional<std::string> extract_answer(const std::string& text) {
    static const std::regex boxed(R"(\\boxed\{([^}]*)\})");
    static const std::regex final_answer(R"(final\s+answer\s*:\s*([^\n\r]+))",
                                         std::regex_constants::icase);
    static const std::regex answer_phrase(
        R"((?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*(-?\d[\d,]*))",
        std::regex_constants::icase);
    static const std::regex discourse_quantity(
        R"((?:therefore|thus|hence|so),?\s+(?:the\s+)?(?:answer|area|sum|difference|product|remainder|probability|count|number|value|total)[^\n\r]{0,64}?(?:is|=)\s*(-?\d[\d,]*))",
        std::regex_constants::icase);
    static const std::regex m_plus_n(R"(m\s*\+\s*n\s*=\s*(-?\d[\d,]*))",
                                     std::regex_constants::icase);
    static const std::regex integer("-?\\d[\\d,]*");

    std::smatch match;
    if (std::regex_search(text, match, boxed)) {
        if (auto answer = detail::normalize_integer(match.str(1)))
            return answer;
    }
    if (std::regex_search(text, match, final_answer)) {
        if (auto answer = detail::normalize_integer(match.str(1)))
            return answer;
    }

    std::optional<std::string> answer;
    detail::capture_last_phrase(text, answer_phrase, answer);
    detail::capture_last_phrase(text, discourse_quantity, answer);
    detail::capture_last_phrase(text, m_plus_n, answer);
    if (answer)
        return answer;

    for (std::sregex_iterator iterator(text.begin(), text.end(), integer), end; iterator != end;
         ++iterator) {
        answer = detail::normalize_integer(iterator->str(0));
    }
    return answer;
}

} // namespace trtmc::examples::dataset_benchmark
