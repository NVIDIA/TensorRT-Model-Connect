/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/eagle3.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace trtmc::llama::speculative {
namespace {
std::vector<std::int32_t> chain(int rows) {
    std::vector<std::int32_t> parents(rows);
    std::iota(parents.begin(), parents.end(), -1);
    return parents;
}

template <typename T>
std::vector<T> slice(const std::vector<T>& source, std::size_t start, std::size_t end) {
    if (start > end || end > source.size())
        throw std::out_of_range("EAGLE3 feature slice");
    return {source.begin() + start, source.begin() + end};
}
} // namespace

Eagle3::Eagle3(Engine& draft, const std::vector<std::int32_t>& mapping, int depth, int width)
    : draft_(draft), mapping_(mapping), depth_(depth), width_(width) {
    if (width < 1 || width > 2 || depth < 1 ||
        depth + 1 > draft.contract().query_limit(Phase::kDecode))
        throw std::invalid_argument("EAGLE3 proposal exceeds the compiled query profile");
}

void Eagle3::prefill(const std::vector<std::int32_t>& prompt, int root,
                     FeatureView target_features) {
    const auto& c = draft_.contract();
    auto shifted = slice(prompt, 1, prompt.size());
    shifted.push_back(root);
    for (int start = 0; start < static_cast<int>(prompt.size());) {
        const int rows =
            std::min(c.query_limit(Phase::kPrefill), static_cast<int>(prompt.size()) - start);
        result_ = draft_.run(Phase::kPrefill, slice(shifted, start, start + rows), start,
                             chain(rows), false, target_features.slice(start, rows));
        start += rows;
    }
}

CandidateTree Eagle3::propose(int root, int committed, int remaining) {
    const auto& c = draft_.contract();
    // The root was emitted but is pending in the target cache. The draft has
    // consumed (target_feature[t-1], embedding[root]) at draft position t-1.
    const int depth = std::min({depth_, remaining, (c.capacity - committed - 1) / width_});
    CandidateTree tree{{root}, {-1}};
    int predecessor = 0;
    for (int step = 0; step < depth; ++step) {
        const int best = result_.token();
        const int candidate = mapping_[best];
        const int best_row = static_cast<int>(tree.tokens.size());
        tree.tokens.push_back(candidate);
        tree.parents.push_back(predecessor);
        if (width_ == 2) {
            // Expose a sibling at each depth and expand the best branch.
            // Full beam scoring and pruning are a separate future policy.
            const int second = result_.token(0, 1);
            tree.tokens.push_back(mapping_[second]);
            tree.parents.push_back(predecessor);
        }
        predecessor = best_row;
        if (step + 1 < depth) {
            const auto recurrent = result_.features.slice(result_.features.rows - 1, 1);
            result_ = draft_.run(Phase::kDecode, {candidate}, committed + step, {-1}, false, {},
                                 recurrent);
        }
    }
    return tree;
}

void Eagle3::feedback(const CandidateTree& tree, const std::vector<std::int32_t>& path,
                      FeatureView target_features, int bonus, int committed) {
    std::vector<std::int32_t> tokens;
    // Replace recurrent proposals with verified target features. Pair each
    // feature with the following accepted token, ending with the pending bonus.
    for (std::size_t index = 0; index < path.size(); ++index) {
        tokens.push_back(index + 1 < path.size() ? tree.tokens[path[index + 1]] : bonus);
    }
    result_ = draft_.run(Phase::kDecode, tokens, committed, chain(static_cast<int>(tokens.size())),
                         false, target_features, {}, path);
}

} // namespace trtmc::llama::speculative
