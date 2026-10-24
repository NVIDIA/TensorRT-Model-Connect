/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/trtmc.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument("expected bundle runtime_root text");
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        if (model.info().family != "nomic_bert")
            throw std::runtime_error("expected Nomic bundle");
        auto task = model.task<trtmc::TextToEmbedding>();
        auto first = task.run({argv[3], trtmc::EmbeddingRole::Default});
        std::cout << std::setprecision(9) << "[";
        for (auto role : {trtmc::EmbeddingRole::Default, trtmc::EmbeddingRole::Query,
                          trtmc::EmbeddingRole::Document}) {
            const auto result = task.run({argv[3], role});
            if (result.values().size() != 768 || result.pooling() != "mean" ||
                result.normalization() != "l2")
                throw std::runtime_error("Nomic embedding contract mismatch");
            if (role != trtmc::EmbeddingRole::Default)
                std::cout << ",";
            std::cout << "[";
            for (size_t index = 0; index < result.values().size(); ++index) {
                const float value = result.values()[index];
                if (!std::isfinite(value) ||
                    (role == trtmc::EmbeddingRole::Default && value != first.values()[index]))
                    throw std::runtime_error("Nomic output is non-finite or unstable across calls");
                if (index)
                    std::cout << ",";
                std::cout << value;
            }
            std::cout << "]";
        }
        std::cout << "]\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
