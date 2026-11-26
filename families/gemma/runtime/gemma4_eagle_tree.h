/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/gemma/runtime/gemma4_jetspec_tree.h"

#include <numeric>
#include <utility>

namespace trtmc::gemma {

// Unlike a block draft, EAGLE's next distribution depends on the parent's
// tokens and hidden state. Expand the ten best paths at each depth; only
// afterwards select the sixty best nodes (including the pending root).
template <typename Children>
std::vector<JetSpecNode> eagle_tree(int32_t anchor, int32_t depth_limit, Children children) {
    if (depth_limit < 0 || depth_limit > 6)
        throw std::invalid_argument("Invalid Gemma EAGLE3 tree depth");
    std::vector<JetSpecNode> nodes{{-1, anchor, 0, 0.0f}};
    const auto higher = [&](int32_t a, int32_t b) {
        return nodes[a].log_probability != nodes[b].log_probability
                   ? nodes[a].log_probability > nodes[b].log_probability
                   : a < b;
    };
    std::vector<int32_t> frontier{0};
    for (int32_t depth = 0; depth < depth_limit; ++depth) {
        std::vector<int32_t> next;
        for (const auto parent : frontier) {
            std::vector<int32_t> path(depth);
            auto cur = parent;
            for (int i = depth; i > 0; --i) {
                path[i - 1] = nodes[cur].token;
                cur = nodes[cur].parent;
            }
            const auto siblings = children(path);
            if (siblings.size() != 10)
                throw std::runtime_error("Gemma EAGLE3 requires ten children per scored parent");
            for (const auto& [token, log_prob] : siblings) {
                if (!std::isfinite(log_prob) || log_prob > 0)
                    throw std::runtime_error("Invalid Gemma EAGLE3 branch log probability");
                next.push_back(static_cast<int32_t>(nodes.size()));
                nodes.push_back(
                    {parent, token, depth + 1, nodes[parent].log_probability + log_prob});
            }
        }
        std::partial_sort(next.begin(), next.begin() + 10, next.end(), higher);
        next.resize(10);
        frontier = std::move(next);
    }
    std::vector<int32_t> selected(nodes.size());
    std::iota(selected.begin(), selected.end(), 0);
    auto keep = std::min<size_t>(60, selected.size());
    std::partial_sort(selected.begin(), selected.begin() + keep, selected.end(), higher);
    selected.resize(keep);
    // Restore ancestor-first order for the verifier. A child cannot outrank
    // its own parent, including when its normalized probability rounds to 1.
    std::sort(selected.begin(), selected.end());
    std::vector<JetSpecNode> tree;
    std::vector<int32_t> index(nodes.size(), -1);
    for (auto id : selected) {
        auto node = nodes[id];
        if (id && (node.parent = index[node.parent]) < 0)
            throw std::runtime_error("Gemma EAGLE3 selection omitted a predecessor");
        index[id] = static_cast<int32_t>(tree.size());
        tree.push_back(node);
    }
    return tree;
}

} // namespace trtmc::gemma
