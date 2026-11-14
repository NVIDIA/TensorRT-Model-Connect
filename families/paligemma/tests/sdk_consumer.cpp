/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++ SDK consumer driven by the family E2E: every request goes through
// the images_text_to_text Task of a loaded bundle.
//
//   test_paligemma_sdk_cpp BUNDLE RUNTIME_ROOT REQUESTS.json OUTPUT.json
//
// REQUESTS.json lists {"prompt", "images": [{"path", "height", "width"}], "max_new_tokens"};
// each image file holds interleaved 8-bit RGB rows.
#include "trtmc/language.hpp"

#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr
            << "usage: test_paligemma_sdk_cpp BUNDLE RUNTIME_ROOT REQUESTS.json OUTPUT.json\n";
        return 2;
    }
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        const auto model = trtmc::Model::load(argv[1], options);
        const auto task = model.task<trtmc::ImagesTextToText>();
        std::ifstream input(argv[3]);
        const auto requests = nlohmann::json::parse(input);
        nlohmann::json output = nlohmann::json::array();
        for (const auto& request : requests) {
            std::deque<std::vector<std::uint8_t>> pixels;
            std::vector<trtmc::ImagesTextPart> parts;
            for (const auto& image : request.at("images")) {
                std::ifstream file(image.at("path").get<std::string>(), std::ios::binary);
                if (!file)
                    throw std::runtime_error("cannot read RGB image");
                pixels.emplace_back(std::istreambuf_iterator<char>(file),
                                    std::istreambuf_iterator<char>());
                parts.emplace_back(trtmc::ImageInput{{pixels.back().data(), pixels.back().size()},
                                                     image.at("height").get<std::uint32_t>(),
                                                     image.at("width").get<std::uint32_t>()});
            }
            parts.emplace_back(trtmc::TextPart{request.at("prompt").get<std::string>()});
            const trtmc::Config config{
                {"max_new_tokens", request.at("max_new_tokens").get<std::int64_t>()}};
            const auto result =
                task.run(trtmc::ImagesTextToTextRequest::from_parts(std::move(parts)), config);
            const auto ids = result.token_ids();
            output.push_back({{"text", std::string(result.text())},
                              {"token_ids", std::vector<std::int32_t>(ids.begin(), ids.end())},
                              {"prefill_ms", result.prefill_ms()},
                              {"decode_ms", result.decode_ms()}});
            std::cout << output.back().dump() << '\n';
        }
        std::ofstream(argv[4]) << output.dump(2);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
