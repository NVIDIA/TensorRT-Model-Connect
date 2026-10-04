/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public SDK consumer for the family-owned end-to-end test. */
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <trtmc/features.hpp>

namespace {
void json_string(std::string_view value) {
    std::cout << '"';
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\')
            std::cout << '\\' << static_cast<char>(byte);
        else if (byte < 32)
            std::cout << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                      << static_cast<unsigned>(byte) << std::dec << std::setfill(' ');
        else
            std::cout << static_cast<char>(byte);
    }
    std::cout << '"';
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " BUNDLE RUNTIME_ROOT TEXT\n";
        return 2;
    }
    try {
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::TextToPooledFeatures>();
            if (!task.config_fields().empty())
                throw std::runtime_error("XLNet must expose no runtime Config fields");
            return task.run({trtmc::TextSource{std::string(argv[3])}});
        }();
        const auto values = result.values();
        if (values.empty())
            throw std::runtime_error("XLNet must provide a nonempty pooled feature vector");
        for (const auto value : values) {
            if (!std::isfinite(value))
                throw std::runtime_error("pooled features contain a nonfinite value");
        }
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"task\":\"text_to_pooled_features\",\"pooling\":";
        json_string(result.pooling());
        std::cout << ",\"normalization\":";
        json_string(result.normalization());
        std::cout << ",\"dim\":" << values.size() << ",\"values\":[";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i)
                std::cout << ',';
            std::cout << values[i];
        }
        std::cout << "]}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write pooled-features output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
