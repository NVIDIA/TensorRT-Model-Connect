/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/muse_glimmer/runtime/dflash2_tree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace trtmc::muse_glimmer {
VerifyTree dflash2_tree(int32_t anchor, const std::vector<int32_t>& ids, const SelectRow& select) {
    constexpr int k = 16;
    if (ids.size() != 15 * k)
        throw std::invalid_argument("DFlash2 DDTree requires fifteen selector rows of width 16");
    // Each child distribution is conditional on its own predecessor, not the
    // greedy spine's token. Repeated (row, predecessor) states share scores.
    using Children = std::array<std::pair<int32_t, float>, 4>;
    std::map<std::pair<int32_t, int32_t>, Children> cache;
    struct Prefix {
        int32_t parent, token, depth, slot;
        float logp;
        bool operator<(const Prefix& rhs) const {
            if (logp != rhs.logp)
                return logp < rhs.logp;
            if (parent != rhs.parent)
                return parent > rhs.parent;
            return slot > rhs.slot;
        }
    };
    std::priority_queue<Prefix> frontier;
    auto expand = [&](int32_t parent, int32_t token, int32_t depth, float logp) {
        if (depth == 15)
            return;
        auto [it, fresh] = cache.try_emplace({depth, token});
        if (fresh) {
            const auto candidates = ids.begin() + depth * k;
            const auto scores = select(depth, token);
            if (scores.size() != k || !std::all_of(scores.begin(), scores.end(),
                                                   [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("Non-finite DFlash2 selector tree score");
            std::vector<int32_t> slots(k);
            std::iota(slots.begin(), slots.end(), 0);
            std::partial_sort(slots.begin(), slots.begin() + 4, slots.end(), [&](int a, int b) {
                return scores[a] == scores[b] ? candidates[a] < candidates[b]
                                              : scores[a] > scores[b];
            });
            const float max = scores[slots[0]];
            float mass = 0;
            for (float score : scores)
                mass += std::exp(score - max);
            const float normalizer = std::log(mass);
            for (int i = 0; i < 4; ++i) {
                const int slot = slots[i];
                it->second[i] = {candidates[slot], (scores[slot] - max) - normalizer};
            }
        }
        for (int i = 0; i < 4; ++i) {
            const auto& child = it->second[i];
            frontier.push({parent, child.first, depth + 1, i, logp + child.second});
        }
    };
    VerifyTree tree{{anchor, -1}};
    expand(0, anchor, 0, 0);
    while (tree.size() < 32 && !frontier.empty()) {
        const auto node = frontier.top();
        frontier.pop();
        tree.push_back({node.token, node.parent});
        if (tree.size() < 32)
            expand(static_cast<int32_t>(tree.size()) - 1, node.token, node.depth, node.logp);
    }
    return tree;
}
} // namespace trtmc::muse_glimmer
