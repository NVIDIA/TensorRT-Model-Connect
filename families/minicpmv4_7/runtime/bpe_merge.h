/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

namespace trtmc::minicpmv4_7 {

// One BPE merge: its rank (index in the merge list, lower merges first) and the merged token's id.
struct BpeMergeRule {
    std::int32_t rank;
    std::int32_t id;
};

// Merges the token ids of one pre-token in the order of Hugging Face tokenizers' BPE model: a
// min-heap of adjacent pairs keyed by (rank, position) over a linked list of symbols; an entry
// whose pair changed after it was queued is skipped. O(n log n) for n symbols.
// lookup(left, right) returns the rule of an adjacent pair, or nullptr when the pair has none.
template <typename Lookup>
std::vector<std::int32_t> merge_bpe(std::vector<std::int32_t> ids, const Lookup& lookup) {
    struct Candidate {
        std::int32_t rank;
        std::size_t position;
        std::int32_t id;
        bool operator>(const Candidate& other) const {
            return rank != other.rank ? rank > other.rank : position > other.position;
        }
    };
    constexpr std::size_t kNone = static_cast<std::size_t>(-1);
    const std::size_t size = ids.size();
    std::vector<std::size_t> previous(size);
    std::vector<std::size_t> next(size);
    std::vector<bool> removed(size, false);
    for (std::size_t index = 0; index < size; ++index) {
        previous[index] = index == 0 ? kNone : index - 1;
        next[index] = index + 1 == size ? kNone : index + 1;
    }
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;
    const auto enqueue = [&](std::size_t left) {
        if (left == kNone || next[left] == kNone)
            return;
        if (const BpeMergeRule* rule = lookup(ids[left], ids[next[left]]))
            queue.push({rule->rank, left, rule->id});
    };
    for (std::size_t index = 0; index + 1 < size; ++index)
        enqueue(index);
    while (!queue.empty()) {
        const Candidate top = queue.top();
        queue.pop();
        if (removed[top.position] || next[top.position] == kNone)
            continue;
        const std::size_t right = next[top.position];
        const BpeMergeRule* current = lookup(ids[top.position], ids[right]);
        if (current == nullptr || current->id != top.id)
            continue;
        ids[top.position] = top.id;
        removed[right] = true;
        next[top.position] = next[right];
        if (next[right] != kNone)
            previous[next[right]] = top.position;
        enqueue(previous[top.position]);
        enqueue(top.position);
    }
    std::vector<std::int32_t> result;
    result.reserve(size);
    for (std::size_t index = 0; index < size; ++index) {
        if (!removed[index])
            result.push_back(ids[index]);
    }
    return result;
}

} // namespace trtmc::minicpmv4_7
