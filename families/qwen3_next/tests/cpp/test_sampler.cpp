/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_next/runtime/sampler.h"

#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

int main() {
    try {
        const std::array<float, 3> logits{1.0F, 0.9F, 0.8F};
        for (const auto top_k : {0, -1}) {
            trtmc::Qwen3NextSamplingParams params;
            params.top_k = top_k;
            auto sampler = trtmc::create_qwen3_next_sampler(params);
            std::vector<int32_t> sequence;
            std::set<int32_t> observed;
            for (int i = 0; i < 128; ++i) {
                const auto id = sampler->sample(logits.data(), logits.size(), params).token_id;
                sequence.push_back(id);
                observed.insert(id);
            }
            if (observed.size() != logits.size())
                throw std::runtime_error("unlimited top-k must sample the full vocabulary");
            sampler->reset();
            for (const auto id : sequence)
                if (sampler->sample(logits.data(), logits.size(), params).token_id != id)
                    throw std::runtime_error("default sampling must replay after reset");
        }
        trtmc::Qwen3NextSamplingParams greedy;
        auto sampler = trtmc::create_qwen3_next_sampler(greedy);
        for (int i = 0; i < 16; ++i)
            if (sampler->sample(logits.data(), logits.size(), greedy).token_id != 0)
                throw std::runtime_error("top-k one must preserve greedy decoding");
        std::cout << "Qwen3-Next sampler contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
