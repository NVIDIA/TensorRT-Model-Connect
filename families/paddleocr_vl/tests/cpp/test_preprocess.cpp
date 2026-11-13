/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/paddleocr_vl/runtime/preprocess.h"
#include "families/paddleocr_vl/runtime/tokenizer.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

int main() {
    nlohmann::json input;
    std::cin >> input;
    const int32_t height = input.at("height"), width = input.at("width");
    const int32_t ph = input.at("profile_height"), pw = input.at("profile_width");
    const int32_t sequence = ph / 28 * (pw / 28);
    std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3);
    std::ifstream source(input.at("source").get<std::string>(), std::ios::binary);
    source.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(float));
    if (!source) return 2;
    const auto image = trtmc::prepare_paddleocr_image(pixels.data(), height, width, ph, pw,
        input.at("min_pixels"), input.at("max_pixels"));
    std::ofstream output(input.at("output").get<std::string>(), std::ios::binary);
    output.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(float));
    std::ifstream tokenizer_file(input.at("tokenizer").get<std::string>());
    const std::string tokenizer_json((std::istreambuf_iterator<char>(tokenizer_file)), {});
    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    std::string image_tokens;
    for (int32_t i = 0; i < sequence; ++i) image_tokens += input.at("image_token").get<std::string>();
    const auto prompt = input.at("prefix").get<std::string>() + image_tokens +
        input.at("between").get<std::string>() + input.at("question").get<std::string>() +
        input.at("suffix").get<std::string>();
    const auto ids = tokenizer->encode(prompt);
    const auto positions = trtmc::paddleocr_position_ids(ids, input.at("image_id"), sequence,
        ph / 28, pw / 28, input.at("profile_length"));
    std::cout << nlohmann::json{{"ids", ids}, {"positions", positions}}.dump() << '\n';
}
