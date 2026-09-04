/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

struct CanarySegmentSpan {
    int64_t offset{0};
    int32_t count{0};
};

inline int64_t canary_seconds_to_samples(float seconds, int32_t sample_rate,
                                         const char* option_name) {
    const double samples = static_cast<double>(seconds) * static_cast<double>(sample_rate);
    if (!std::isfinite(samples) || samples < 0.0 ||
        samples > static_cast<double>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument(std::string("Canary ") + option_name +
                                    " produces an invalid sample count");
    }
    return static_cast<int64_t>(std::llround(samples));
}

inline void validate_canary_segment_audio_dimensions(int32_t num_samples, int32_t sample_rate) {
    if (num_samples <= 0)
        throw std::invalid_argument("Canary segment planning requires positive audio dimensions");
    if (sample_rate <= 0)
        throw std::invalid_argument("Canary segment planning requires positive audio dimensions");
}

inline void validate_canary_segment_sample_bounds(int64_t max_samples, int64_t min_samples,
                                                  int64_t overlap_samples) {
    if (max_samples <= 0)
        throw std::invalid_argument("Canary segment duration/overlap configuration is invalid");
    if (min_samples > max_samples)
        throw std::invalid_argument("Canary segment duration/overlap configuration is invalid");
    if (overlap_samples >= (min_samples > 0 ? min_samples : max_samples)) {
        throw std::invalid_argument("Canary segment duration/overlap configuration is invalid");
    }
}

inline std::vector<CanarySegmentSpan>
plan_canary_fixed_segments(int64_t input_samples, int64_t max_samples, int64_t overlap_samples) {
    std::vector<CanarySegmentSpan> spans;
    const int64_t stride = max_samples - overlap_samples;
    for (int64_t offset = 0; offset < input_samples; offset += stride) {
        const int64_t count = std::min(max_samples, input_samples - offset);
        spans.push_back({offset, static_cast<int32_t>(count)});
        if (offset + count == input_samples)
            break;
    }
    return spans;
}

inline std::vector<CanarySegmentSpan> plan_canary_dynamic_segments(int64_t input_samples,
                                                                   int64_t max_samples,
                                                                   int64_t min_samples,
                                                                   int64_t overlap_samples) {
    // Use the fewest windows that can cover the signal at the requested
    // overlap. Equalize their lengths to avoid a heavily padded tail window.
    const int64_t stride_cap = max_samples - overlap_samples;
    const int64_t segment_count =
        std::max<int64_t>(2, (input_samples - overlap_samples + stride_cap - 1) / stride_cap);
    int64_t window_samples =
        (input_samples + (segment_count - 1) * overlap_samples + segment_count - 1) / segment_count;
    window_samples = std::max(window_samples, min_samples);
    if (window_samples > max_samples) {
        throw std::invalid_argument(
            "Canary cannot cover the input with the requested dynamic segment bounds");
    }

    const int64_t offset_span = input_samples - window_samples;
    std::vector<CanarySegmentSpan> spans;
    spans.reserve(static_cast<std::size_t>(segment_count));
    for (int64_t index = 0; index < segment_count; ++index) {
        const int64_t offset =
            (index * offset_span + (segment_count - 1) / 2) / (segment_count - 1);
        spans.push_back({offset, static_cast<int32_t>(window_samples)});
    }
    return spans;
}

inline std::vector<CanarySegmentSpan> plan_canary_segments(int32_t num_samples, int32_t sample_rate,
                                                           float max_segment_seconds,
                                                           float min_segment_seconds,
                                                           float overlap_seconds) {
    validate_canary_segment_audio_dimensions(num_samples, sample_rate);
    const int64_t max_samples =
        canary_seconds_to_samples(max_segment_seconds, sample_rate, "segment duration");
    const int64_t min_samples = min_segment_seconds > 0.0F
                                    ? canary_seconds_to_samples(min_segment_seconds, sample_rate,
                                                                "minimum segment duration")
                                    : 0;
    const int64_t overlap_samples =
        canary_seconds_to_samples(overlap_seconds, sample_rate, "segment overlap");
    validate_canary_segment_sample_bounds(max_samples, min_samples, overlap_samples);

    const int64_t input_samples = num_samples;
    if (input_samples <= max_samples)
        return {{0, num_samples}};
    if (min_samples == 0)
        return plan_canary_fixed_segments(input_samples, max_samples, overlap_samples);
    return plan_canary_dynamic_segments(input_samples, max_samples, min_samples, overlap_samples);
}

template <typename T>
inline std::vector<uint16_t>
build_canary_lcs_lengths(const std::vector<T>& left, const std::vector<T>& right,
                         std::size_t left_begin, std::size_t right_size) {
    const std::size_t left_size = left.size() - left_begin;
    std::vector<uint16_t> lengths((left_size + 1) * (right_size + 1), 0);
    const auto at = [&lengths, right_size](std::size_t i, std::size_t j) -> uint16_t& {
        return lengths[i * (right_size + 1) + j];
    };
    for (std::size_t i = 1; i <= left_size; ++i) {
        for (std::size_t j = 1; j <= right_size; ++j) {
            if (left[left_begin + i - 1] == right[j - 1]) {
                at(i, j) = static_cast<uint16_t>(at(i - 1, j - 1) + 1);
            } else {
                at(i, j) = std::max(at(i - 1, j), at(i, j - 1));
            }
        }
    }
    return lengths;
}

template <typename T>
inline std::vector<std::pair<std::size_t, std::size_t>>
backtrack_canary_lcs_matches(const std::vector<T>& left, const std::vector<T>& right,
                             std::size_t left_begin, std::size_t right_size,
                             const std::vector<uint16_t>& lengths) {
    const std::size_t left_size = left.size() - left_begin;
    const auto at = [&lengths, right_size](std::size_t i, std::size_t j) {
        return lengths[i * (right_size + 1) + j];
    };
    std::vector<std::pair<std::size_t, std::size_t>> matches;
    std::size_t i = left_size;
    std::size_t j = right_size;
    while (i > 0 && j > 0) {
        if (left[left_begin + i - 1] == right[j - 1]) {
            matches.emplace_back(left_begin + i - 1, j - 1);
            --i;
            --j;
        } else if (at(i - 1, j) >= at(i, j - 1)) {
            --i;
        } else {
            --j;
        }
    }
    std::reverse(matches.begin(), matches.end());
    return matches;
}

inline bool
canary_lcs_matches_boundary(const std::vector<std::pair<std::size_t, std::size_t>>& matches,
                            std::size_t left_size, std::size_t minimum_matches) {
    if (matches.empty() || matches.size() < minimum_matches)
        return false;
    const std::size_t boundary_slack = std::max<std::size_t>(8, matches.size());
    if (matches.front().second > boundary_slack)
        return false;
    if (left_size - 1 - matches.back().first > boundary_slack)
        return false;
    const std::size_t left_overlap = left_size - matches.front().first;
    const std::size_t right_overlap = matches.back().second + 1;
    return matches.size() * 3 >= std::min(left_overlap, right_overlap);
}

template <typename T>
inline std::vector<T>
merge_canary_lcs_boundary(const std::vector<T>& left, const std::vector<T>& right,
                          std::size_t boundary_window = 32, std::size_t minimum_matches = 2) {
    if (left.empty())
        return right;
    if (right.empty())
        return left;

    const std::size_t left_begin =
        left.size() > boundary_window ? left.size() - boundary_window : 0;
    const std::size_t right_size = std::min(right.size(), boundary_window);
    const auto lengths = build_canary_lcs_lengths(left, right, left_begin, right_size);
    const auto matches = backtrack_canary_lcs_matches(left, right, left_begin, right_size, lengths);
    if (!canary_lcs_matches_boundary(matches, left.size(), minimum_matches))
        return {};

    const auto [left_pivot, right_pivot] = matches.back();
    std::vector<T> merged;
    merged.reserve(left_pivot + 1 + right.size() - right_pivot - 1);
    merged.insert(merged.end(), left.begin(),
                  left.begin() + static_cast<std::ptrdiff_t>(left_pivot + 1));
    merged.insert(merged.end(), right.begin() + static_cast<std::ptrdiff_t>(right_pivot + 1),
                  right.end());
    return merged;
}

inline std::vector<int32_t> merge_canary_token_segments(const std::vector<int32_t>& left,
                                                        const std::vector<int32_t>& right,
                                                        int32_t eot_token_id) {
    auto content = [eot_token_id](const std::vector<int32_t>& tokens) {
        auto end = tokens.end();
        while (end != tokens.begin() && *(end - 1) == eot_token_id)
            --end;
        return std::vector<int32_t>(tokens.begin(), end);
    };
    const auto left_content = content(left);
    const auto right_content = content(right);
    auto merged = merge_canary_lcs_boundary(left_content, right_content);
    if (merged.empty()) {
        merged = left_content;
        merged.insert(merged.end(), right_content.begin(), right_content.end());
    }
    if ((!left.empty() && left.back() == eot_token_id) ||
        (!right.empty() && right.back() == eot_token_id)) {
        merged.push_back(eot_token_id);
    }
    return merged;
}

struct CanaryBoundaryWord {
    std::string original;
    std::string normalized;

    bool operator==(const CanaryBoundaryWord& other) const {
        return normalized == other.normalized;
    }
};

inline std::vector<CanaryBoundaryWord> split_canary_boundary_words(const std::string& text) {
    std::istringstream input(text);
    std::vector<CanaryBoundaryWord> words;
    std::string word;
    while (input >> word) {
        std::string normalized = word;
        while (!normalized.empty() &&
               std::ispunct(static_cast<unsigned char>(normalized.front())) != 0) {
            normalized.erase(normalized.begin());
        }
        while (!normalized.empty() &&
               std::ispunct(static_cast<unsigned char>(normalized.back())) != 0) {
            normalized.pop_back();
        }
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (!normalized.empty())
            words.push_back({std::move(word), std::move(normalized)});
    }
    return words;
}

inline std::string merge_canary_text_segments(const std::string& left, const std::string& right) {
    if (left.empty())
        return right;
    if (right.empty())
        return left;
    const auto left_words = split_canary_boundary_words(left);
    const auto right_words = split_canary_boundary_words(right);
    constexpr std::size_t CanaryMaxMergedBoundaryWords = 12;
    auto merged = merge_canary_lcs_boundary(left_words, right_words, 16, 1);
    // At conversational speech rates, a two-second overlap is normally no
    // more than about twelve words. A larger deletion is more likely an
    // ambiguous match against repeated boilerplate than the actual boundary.
    const std::size_t removed_words =
        merged.empty() ? 0 : left_words.size() + right_words.size() - merged.size();
    if (merged.empty() || removed_words > CanaryMaxMergedBoundaryWords)
        return left + ' ' + right;

    std::ostringstream output;
    for (std::size_t index = 0; index < merged.size(); ++index) {
        if (index > 0)
            output << ' ';
        output << merged[index].original;
    }
    return output.str();
}

} // namespace trtmc
