/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_8_flash_next/runtime/ngram.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace trtmc::qwen3_8_flash_next;

int main() {
    NgramLayout layout{{0x8000000000000001ULL, 3, 5}, {}, {}, 7, 32};
    layout.sizes.fill(11);
    for (std::size_t h = 0; h < layout.offsets.size(); ++h)
        layout.offsets[h] = static_cast<std::int64_t>(h) * 11;
    NgramLookup lookup(layout, 176);

    // Reference table indices from signed int64 wrapped products, XOR, and
    // nonnegative remainder. The EOS in the middle starts a new trigram window.
    constexpr std::array<int, 4> bigram{1, 4, 3, 0}, trigram{8, 4, 1, 2};
    const auto whole = lookup.lookup({2, 7, 3, 1}, lookup.initial_history());
    if (whole.rows.size() != 64)
        return 1;
    for (std::size_t i = 0; i < bigram.size(); ++i) {
        for (std::size_t h = 0; h < layout.sizes.size(); ++h) {
            if (whole.rows[16 * i + h] != layout.offsets[h] + (h < 8 ? bigram[i] : trigram[i])) {
                std::cerr << "Unexpected lexical row at token " << i << ", head " << h << '\n';
                return 1;
            }
        }
    }

    auto first = lookup.lookup({2, 7}, lookup.initial_history());
    auto second = lookup.lookup({3, 1}, first.history);
    first.rows.insert(first.rows.end(), second.rows.begin(), second.rows.end());
    if (whole.rows != first.rows || whole.history != second.history ||
        whole.history != std::array<std::int32_t, 2>{3, 1})
        return 1;

    try {
        lookup.lookup({layout.vocabulary_size}, lookup.initial_history());
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << "Accepted an out-of-vocabulary token\n";
    return 1;
}
