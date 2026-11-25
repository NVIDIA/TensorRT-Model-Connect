/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/bailing_hybrid/runtime/generation.h"
#include "trtmc/core.hpp"

#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc == 1) {
        using trtmc::bailing_hybrid::generation_limit;
        if (generation_limit(std::nullopt, 1, 16) != 15 ||
            generation_limit(std::nullopt, 3, 128) != 32 || generation_limit(4, 1, 16) != 4)
            return 1;
        for (int64_t invalid : {-1, 0, 16}) {
            bool rejected = false;
            try {
                generation_limit(invalid, 1, 16);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            if (!rejected)
                return 1;
        }
        std::cout << "Ling generation defaults fit the loaded profile\n";
        return 0;
    }
    if (argc != 3)
        return 2;
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        const auto model = trtmc::Model::load(argv[1], options);
        const auto task = model.task<trtmc::TextContinuation>();
        nlohmann::json requests;
        std::cin >> requests;
        auto results = nlohmann::json::array();
        for (const auto& request : requests) {
            trtmc::Config config;
            if (request.contains("max_new_tokens"))
                config.add("max_new_tokens", request.at("max_new_tokens").get<int64_t>());
            const auto result = task.run({request.at("prompt").get<std::string>()}, config);
            results.push_back({{"text", result.text()},
                               {"token_ids", std::vector<int32_t>(result.token_ids().begin(),
                                                                  result.token_ids().end())}});
        }
        std::cout << results.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
