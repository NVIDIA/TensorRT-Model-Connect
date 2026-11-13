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
        if (model.info().family != "qwen")
            throw std::runtime_error("expected qwen bundle");
        auto task = model.task<trtmc::TextToEmbedding>();
        const auto result = task.run({argv[3], trtmc::EmbeddingRole::Default});
        if (result.values().size() != 1024 || result.pooling() != "last_token" ||
            result.normalization() != "l2")
            throw std::runtime_error("embedding contract mismatch");
        std::cout << std::setprecision(9) << "[";
        bool first = true;
        for (float value : result.values()) {
            if (!std::isfinite(value))
                throw std::runtime_error("non-finite embedding");
            if (!first)
                std::cout << ",";
            first = false;
            std::cout << value;
        }
        std::cout << "]\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
