/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/trtmc.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument("expected BUNDLE RUNTIME_ROOT REQUEST.json OUTPUT.json");
        nlohmann::json request;
        std::ifstream(argv[3]) >> request;
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        if (model.info().family != "diffusion_gemma")
            throw std::runtime_error("expected the diffusion_gemma model family");
        trtmc::TextContinuationRequest input;
        if (request.contains("input_ids"))
            input.prefix = request.at("input_ids").get<std::vector<std::int32_t>>();
        else
            input.prefix = request.at("prefix").get<std::string>();
        trtmc::Config config{{"max_new_tokens", request.value("max_new_tokens", std::int64_t{256})},
                             {"seed", request.value("seed", std::int64_t{0})}};
        const auto result = model.task<trtmc::TextContinuation>().run(input, config);
        const auto ids = result.token_ids();
        nlohmann::json output{{"text", std::string(result.text())},
                              {"token_ids", std::vector<std::int32_t>(ids.begin(), ids.end())},
                              {"prefill_ms", result.prefill_ms()},
                              {"decode_ms", result.decode_ms()}};
        std::ofstream stream(argv[4]);
        stream << output.dump(2) << '\n';
        stream.close();
        if (!stream)
            throw std::runtime_error("cannot write diffusion_gemma generation result");
        std::cout << output.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
