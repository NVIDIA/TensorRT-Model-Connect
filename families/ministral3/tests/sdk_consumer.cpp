/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++ SDK consumer for the family-owned end-to-end test:
// BUNDLE RUNTIME_ROOT ROLE TEXT, ROLE = default | query | document.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <trtmc/features.hpp>

int main(int argc, char** argv) {
    if (argc != 5)
        return 2;
    try {
        const std::string role_name = argv[3];
        const auto role =
            role_name == "query"      ? trtmc::EmbeddingRole::Query
            : role_name == "document" ? trtmc::EmbeddingRole::Document
            : role_name == "default"
                ? trtmc::EmbeddingRole::Default
                : throw std::invalid_argument("ROLE must be default, query or document");
        const auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            const auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::TextToEmbedding>();
            if (!task.config_fields().empty())
                throw std::runtime_error("ministral3 must expose no runtime configuration");
            return task.run({std::string(argv[4]), role});
        }();
        if (result.pooling() != "mean" || result.normalization() != "l2" || result.values().empty())
            throw std::runtime_error("invalid ministral3 embedding");
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"task\":\"text_to_embedding\",\"pooling\":\"mean\","
                     "\"normalization\":\"l2\",\"values\":[";
        const auto values = result.values();
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (!std::isfinite(values[index]))
                throw std::runtime_error("nonfinite ministral3 embedding value");
            if (index)
                std::cout << ',';
            std::cout << values[index];
        }
        std::cout << "]}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write the ministral3 result");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
