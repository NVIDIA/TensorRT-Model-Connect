/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc::gemma {

// Parent indices point into the same root-inclusive tree. Siblings are genuine
// alternatives for one target token, not successive tokens in a linear block.
struct JetSpecNode {
    int32_t parent;
    int32_t token;
    int32_t depth;
    float log_probability;
};

/// Construct the best-first tree from the causal block draft's per-depth logits.
/// Row zero is the pending anchor; only rows 1..block-1 propose future tokens.
inline std::vector<JetSpecNode> jetspec_tree(const std::vector<float>& logits, int32_t vocabulary,
                                             int32_t anchor, int32_t top_k, int32_t verify_budget) {
    if (vocabulary < 2 || top_k < 2 || top_k > vocabulary || anchor < 0 || anchor >= vocabulary ||
        verify_budget < 2 || logits.empty() || logits.size() % vocabulary != 0)
        throw std::invalid_argument("Invalid Gemma JetSpec proposal dimensions");

    const size_t rows = logits.size() / vocabulary;
    struct Candidate {
        int32_t token;
        float log_probability;
    };
    std::vector<std::vector<Candidate>> candidates(rows - 1);
    for (size_t i = 1; i < rows; ++i) {
        const float* row = logits.data() + i * vocabulary;
        auto& best = candidates[i - 1];
        // Normalize over the entire vocabulary, not over the retained top K.
        // Row normalizers matter when comparing partial paths of different lengths.
        float maximum = -std::numeric_limits<float>::infinity();
        for (int32_t token = 0; token < vocabulary; ++token) {
            if (std::isnan(row[token]) || row[token] == std::numeric_limits<float>::infinity())
                throw std::runtime_error("Invalid Gemma JetSpec draft logit");
            maximum = std::max(maximum, row[token]);
            if (std::isfinite(row[token]) && (best.size() < static_cast<size_t>(top_k) ||
                                              row[token] > best.back().log_probability)) {
                // Ascending vocabulary traversal resolves equal logits by lowest token ID.
                auto pos = std::find_if(best.begin(), best.end(), [&](const Candidate& c) {
                    return row[token] > c.log_probability;
                });
                best.insert(pos, {token, row[token]});
                if (best.size() > static_cast<size_t>(top_k))
                    best.pop_back();
            }
        }
        if (best.empty())
            continue; // A fully masked proposal row cannot extend a branch.
        double sum = 0.0;
        for (int32_t token = 0; token < vocabulary; ++token)
            sum += std::exp(static_cast<double>(row[token] - maximum));
        const double log_sum = std::log(sum);
        for (auto& candidate : best)
            candidate.log_probability =
                static_cast<float>((candidate.log_probability - maximum) - log_sum);
    }

    std::vector<JetSpecNode> tree{{-1, anchor, 0, 0.0f}};
    std::vector<size_t> next{0};
    while (tree.size() < static_cast<size_t>(verify_budget)) {
        int32_t parent = -1;
        float score = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < tree.size(); ++i) {
            const auto depth = static_cast<size_t>(tree[i].depth);
            if (depth == candidates.size() || next[i] == candidates[depth].size())
                continue;
            const float child_score =
                tree[i].log_probability + candidates[depth][next[i]].log_probability;
            // On equal cumulative scores the earlier parent wins.
            if (child_score > score) {
                parent = static_cast<int32_t>(i);
                score = child_score;
            }
        }
        if (parent < 0)
            break;
        const auto depth = tree[parent].depth;
        const auto& child = candidates[depth][next[parent]++];
        tree.push_back({parent, child.token, depth + 1, score});
        next.push_back(0);
    }
    return tree;
}

} // namespace trtmc::gemma
