/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU-only probe for the real runtime sampler. The line protocol contains:
// vocab history_count draw_count temperature top_k top_p min_p penalty seed,
// followed by logits, history token IDs, and shared uniform draws.
#include "families/lfm2/runtime/sampler.h"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    int32_t vocab = 0;
    std::size_t history_count = 0, draw_count = 0;
    trtmc::Lfm2SamplingParams params;
    while (std::cin >> vocab >> history_count >> draw_count >> params.temperature >> params.top_k >>
           params.top_p >> params.min_p >> params.repetition_penalty >> params.seed) {
        if (vocab <= 0)
            return 1;
        std::vector<float> logits(static_cast<std::size_t>(vocab));
        std::vector<int32_t> history(history_count);
        std::vector<float> draws(draw_count);
        for (auto& value : logits)
            std::cin >> value;
        for (auto& value : history)
            std::cin >> value;
        for (auto& value : draws)
            std::cin >> value;
        if (!std::cin)
            return 2;
        const auto distribution =
            trtmc::lfm2_sampling_distribution(logits.data(), vocab, params, history);
        std::vector<float> probabilities(static_cast<std::size_t>(vocab), 0.0F);
        for (std::size_t index = 0; index < distribution.indices.size(); ++index)
            probabilities[distribution.indices[index]] = distribution.probabilities[index];
        std::cout << std::setprecision(9) << "{\"probabilities\":[";
        for (std::size_t index = 0; index < probabilities.size(); ++index)
            std::cout << (index ? "," : "") << probabilities[index];
        std::cout << "],\"coupled_ids\":[";
        for (std::size_t index = 0; index < draws.size(); ++index)
            std::cout << (index ? "," : "")
                      << trtmc::lfm2_sample_from_distribution(distribution, draws[index], params)
                             .token_id;
        auto sampler = trtmc::create_lfm2_sampler(params);
        const auto seeded = sampler->sample(logits.data(), vocab, params, history);
        std::cout << "],\"seeded_id\":" << seeded.token_id << "}\n";
    }
    return 0;
}
