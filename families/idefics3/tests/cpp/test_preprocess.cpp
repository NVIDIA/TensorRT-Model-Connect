/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/idefics3/runtime/preprocess.h"
#include "families/idefics3/runtime/tokenizer.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

int main() {
    nlohmann::json input;
    std::cin >> input;
    const int32_t height = input.at("height"), width = input.at("width");
    std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3);
    std::ifstream source(input.at("source").get<std::string>(), std::ios::binary);
    source.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(float));
    if (!source) return 2;
    const auto image = trtmc::preprocess_idefics3_image(pixels.data(), height, width, 2048, 512);
    std::ofstream output(input.at("output").get<std::string>(), std::ios::binary);
    for (const auto& view : image.views)
        output.write(reinterpret_cast<const char*>(view.data()), view.size() * sizeof(float));
    std::ifstream tokenizer_file(input.at("tokenizer").get<std::string>());
    const std::string tokenizer_json((std::istreambuf_iterator<char>(tokenizer_file)), {});
    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    const auto prompt = input.at("prefix").get<std::string>() +
        trtmc::idefics3_image_prompt(image.rows, image.columns, 64) +
        input.at("between").get<std::string>() + input.at("question").get<std::string>() +
        input.at("suffix").get<std::string>();
    std::cout << nlohmann::json{{"rows", image.rows}, {"columns", image.columns},
        {"views", image.views.size()}, {"ids", tokenizer->encode(prompt)}}.dump() << '\n';
}
