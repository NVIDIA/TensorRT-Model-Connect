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
        if (argc != 5)
            throw std::invalid_argument("expected bundle runtime_root text role");
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        if (model.info().family != "embeddinggemma2")
            throw std::runtime_error("expected embeddinggemma2 bundle");
        trtmc::EmbeddingRole role;
        const std::string mode(argv[4]);
        if (mode == "default")
            role = trtmc::EmbeddingRole::Default;
        else if (mode == "query")
            role = trtmc::EmbeddingRole::Query;
        else if (mode == "document")
            role = trtmc::EmbeddingRole::Document;
        else
            throw std::invalid_argument("invalid role");
        const auto result = model.task<trtmc::TextToEmbedding>().run({argv[3], role});
        if (result.values().size() != 768 || result.pooling() != "mean" ||
            result.normalization() != "l2")
            throw std::runtime_error("embedding contract mismatch");
        std::cout << std::setprecision(9) << "[";
        bool first = true;
        for (const float value : result.values()) {
            if (!std::isfinite(value))
                throw std::runtime_error("non-finite embedding");
            if (!first)
                std::cout << ",";
            first = false;
            std::cout << value;
        }
        std::cout << "]\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
