/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/core.hpp"
#include "trtmc/language.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 8)
        return 2;
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        auto task = model.task<trtmc::ImagesTextToText>();
        std::ifstream image(argv[3], std::ios::binary);
        if (!image)
            throw std::runtime_error("RGB fixture could not be opened");
        std::vector<std::uint8_t> pixels{std::istreambuf_iterator<char>(image), {}};
        const auto height = static_cast<std::uint32_t>(std::stoul(argv[4]));
        const auto width = static_cast<std::uint32_t>(std::stoul(argv[5]));
        const auto request = trtmc::ImagesTextToTextRequest::from_parts({
            trtmc::ImageInput({pixels.data(), pixels.size()}, height, width),
            trtmc::TextPart{argv[6]},
        });
        const trtmc::Config config{
            {"max_new_tokens", std::int64_t{std::stoi(argv[7])}},
            {"top_k", std::int64_t{1}},
        };
        auto result = task.run(request, config);
        const auto ids = result.token_ids();
        const auto text = result.text();
        std::cout << text.size() << '\n';
        std::cout.write(text.data(), static_cast<std::streamsize>(text.size()));
        std::cout << '\n';
        for (const auto id : ids)
            std::cout << id << ' ';
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
