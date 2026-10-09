/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <trtmc/features.hpp>

int main(int argc, char** argv) {
    if (argc != 4)
        return 2;
    try {
        const auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            const auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::TextToPooledFeatures>();
            if (!task.config_fields().empty())
                throw std::runtime_error("unexpected ModernBERT configuration fields");
            return task.run({trtmc::TextSource{std::string(argv[3])}});
        }();
        if (result.pooling() != "cls" || result.normalization() != "none" ||
            result.values().empty())
            throw std::runtime_error("invalid ModernBERT pooled features");
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"task\":\"text_to_pooled_features\",\"pooling\":\"cls\","
                     "\"normalization\":\"none\",\"values\":[";
        const auto values = result.values();
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (!std::isfinite(values[index]))
                throw std::runtime_error("nonfinite ModernBERT pooled feature");
            if (index)
                std::cout << ',';
            std::cout << values[index];
        }
        std::cout << "]}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write ModernBERT result");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
