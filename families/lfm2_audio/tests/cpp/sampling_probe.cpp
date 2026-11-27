/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_audio/runtime/sampling.h"

#include <iomanip>
#include <iostream>
#include <string>

int main() {
    using namespace trtmc::lfm2_audio;
    std::string mode;
    float temperature;
    int top_k;
    std::size_t size;
    while (std::cin >> mode >> temperature >> top_k >> size) {
        std::vector<float> logits(size);
        for (auto& value : logits)
            std::cin >> value;
        double uniform;
        std::cin >> uniform;
        try {
            if (mode == "probabilities") {
                const auto probs = probabilities(logits, temperature, top_k);
                std::cout << std::setprecision(9);
                for (float probability : probs)
                    std::cout << probability << ' ';
                std::cout << '\n';
            } else if (mode == "uniform") {
                std::cout << sample_probabilities(probabilities(logits, temperature, top_k),
                                                  uniform)
                          << '\n';
            } else if (mode == "weights") {
                std::cout << sample_probabilities(logits, uniform) << '\n';
            } else {
                std::mt19937 rng{42};
                std::cout << sample(logits, temperature, top_k, rng) << '\n';
            }
        } catch (const std::invalid_argument&) {
            std::cout << "invalid_argument\n";
        }
    }
}
