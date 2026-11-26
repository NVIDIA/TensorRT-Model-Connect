/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/blip2/runtime/image_preprocess_seam.h"
#include "families/blip2/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
int main() {
    nlohmann::json input;
    std::cin >> input;
    const int32_t h = input.at("height"), w = input.at("width");
    std::vector<float> rgb(static_cast<std::size_t>(h) * w * 3);
    std::ifstream source(input.at("source").get<std::string>(), std::ios::binary);
    source.read(reinterpret_cast<char*>(rgb.data()), rgb.size() * sizeof(float));
    if (!source)
        return 2;
    auto pixels = trtmc::preprocess_blip2_image(rgb.data(), h, w, {});
    std::ofstream output(input.at("output").get<std::string>(), std::ios::binary);
    output.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(float));
    std::ifstream file(input.at("tokenizer").get<std::string>());
    const std::string data((std::istreambuf_iterator<char>(file)), {});
    auto tokenizer = trtmc::CreateBpeTokenizer(data.data(), data.size(), false);
    std::vector<int32_t> ids(32, input.at("image_id").get<int32_t>());
    ids.push_back(2);
    auto text = tokenizer->encode(input.at("question").get<std::string>());
    ids.insert(ids.end(), text.begin(), text.end());
    std::cout << nlohmann::json{{"ids", ids}}.dump() << '\n';
}
