/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C++ SDK consumer for the family-owned end-to-end test.
 *
 *   BUNDLE RUNTIME_ROOT RGB_F32 HEIGHT WIDTH PROMPT MAX_NEW_TOKENS
 *
 * Runs images_text_to_text on one user message (image, then text) and prints
 * {"text": ..., "token_ids": [...]} as JSON.
 */
#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <trtmc/language.hpp>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLE RUNTIME_ROOT RGB_F32 HEIGHT WIDTH PROMPT MAX_NEW_TOKENS\n";
        return 2;
    }
    try {
        const auto height = std::stoul(argv[4]);
        const auto width = std::stoul(argv[5]);
        const auto limit = std::stoll(argv[7]);
        if (!height || !width || height > 65535 || width > 65535 || limit <= 0)
            throw std::invalid_argument("invalid image dimensions or token limit");
        std::vector<float> pixels(height * width * 3);
        std::ifstream input(argv[3], std::ios::binary | std::ios::ate);
        if (!input || static_cast<std::size_t>(input.tellg()) != pixels.size() * sizeof(float))
            throw std::invalid_argument("input must contain HEIGHT x WIDTH x 3 float32 RGB values");
        input.seekg(0);
        input.read(reinterpret_cast<char*>(pixels.data()),
                   static_cast<std::streamsize>(pixels.size() * sizeof(float)));
        if (!input)
            throw std::runtime_error("incomplete RGB input");
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        const auto task = model.task<trtmc::ImagesTextToText>();
        std::vector<trtmc::ImagesTextPart> parts;
        parts.emplace_back(trtmc::ImageInput{{pixels.data(), pixels.size()},
                                             static_cast<std::uint32_t>(height),
                                             static_cast<std::uint32_t>(width)});
        parts.emplace_back(trtmc::TextPart{argv[6]});
        const auto result = task.run(trtmc::ImagesTextToTextRequest::from_parts(std::move(parts)),
                                     {{"max_new_tokens", static_cast<std::int64_t>(limit)}});
        const auto ids = result.token_ids();
        if (ids.empty() || ids.size() > static_cast<std::size_t>(limit))
            throw std::runtime_error(
                "images_text_to_text must return 1..MAX_NEW_TOKENS generated tokens");
        std::cout << nlohmann::json(
                         {{"text", std::string(result.text())},
                          {"token_ids", std::vector<std::int32_t>(ids.begin(), ids.end())}})
                         .dump()
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
