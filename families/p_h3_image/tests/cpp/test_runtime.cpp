/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/image.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        auto task = model.task<trtmc::TextToImage>();
        nlohmann::json requests;
        std::cin >> requests;
        nlohmann::json results = nlohmann::json::array();
        for (const auto& request : requests) {
            std::ifstream input(request.at("noise").get<std::string>(),
                                std::ios::binary | std::ios::ate);
            const auto size = input.tellg();
            if (!input || size <= 0 || size % sizeof(float))
                return 2;
            std::vector<float> noise(static_cast<std::size_t>(size) / sizeof(float));
            input.seekg(0);
            input.read(reinterpret_cast<char*>(noise.data()), size);
            if (!input)
                return 2;
            auto result =
                task.run({request.at("prompt").get<std::string>(), {noise.data(), noise.size()}});
            std::ofstream output(request.at("output").get<std::string>(), std::ios::binary);
            const auto pixels = result.pixels();
            output.write(reinterpret_cast<const char*>(pixels.data()),
                         pixels.size() * sizeof(float));
            if (!output)
                return 2;
            results.push_back({{"height", result.height()},
                               {"width", result.width()},
                               {"channels", result.channels()},
                               {"count", pixels.size()}});
        }
        std::cout << results.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
