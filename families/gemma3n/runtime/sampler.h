/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/gemma3n/runtime/request.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace trtmc::gemma3n {

// Host token selection over float32 logits.
class Sampler {
  public:
    explicit Sampler(const Generation& generation)
        : generation_(generation),
          rng_(generation.seed >= 0 ? static_cast<uint64_t>(generation.seed)
                                    : static_cast<uint64_t>(std::random_device{}())) {}

    int32_t next(const std::vector<float>& logits) {
        const auto vocab = static_cast<int32_t>(logits.size());
        if (generation_.temperature == 0.0)
            return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) -
                                        logits.begin());
        std::vector<int32_t> order(static_cast<std::size_t>(vocab));
        std::iota(order.begin(), order.end(), 0);
        const auto keep = generation_.top_k > 0 ? std::min(generation_.top_k, vocab) : vocab;
        // Ties keep the lower token id first, so the selection is deterministic.
        auto more_likely = [&](int32_t a, int32_t b) {
            return logits[std::size_t(a)] != logits[std::size_t(b)]
                       ? logits[std::size_t(a)] > logits[std::size_t(b)]
                       : a < b;
        };
        std::partial_sort(order.begin(), order.begin() + keep, order.end(), more_likely);
        order.resize(static_cast<std::size_t>(keep));
        std::vector<double> probabilities(order.size());
        const double top = logits[std::size_t(order[0])];
        double total = 0.0;
        for (std::size_t i = 0; i < order.size(); ++i) {
            probabilities[i] =
                std::exp((double(logits[std::size_t(order[i])]) - top) / generation_.temperature);
            total += probabilities[i];
        }
        std::size_t count = 0;
        double mass = 0.0;
        while (count < order.size()) {
            mass += probabilities[count] / total;
            ++count;
            if (mass >= generation_.top_p)
                break;
        }
        std::discrete_distribution<std::size_t> pick(probabilities.begin(),
                                                     probabilities.begin() + std::ptrdiff_t(count));
        return order[pick(rng_)];
    }

  private:
    Generation generation_;
    std::mt19937_64 rng_;
};

} // namespace trtmc::gemma3n
