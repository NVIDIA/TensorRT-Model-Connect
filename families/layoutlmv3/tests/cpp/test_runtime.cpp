/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/features.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    nlohmann::json input;
    std::cin >> input;
    const uint32_t h = input.at("height"), w = input.at("width");
    std::vector<float> pixels(static_cast<size_t>(h) * w * 3);
    std::ifstream file(input.at("image_path").get<std::string>(), std::ios::binary);
    file.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(float));
    if (!file)
        return 2;
    trtmc::DocumentWordsToTokenFeaturesRequest request{
        trtmc::ImageInput({pixels.data(), pixels.size()}, h, w), {}};
    for (const auto& word : input.at("words")) {
        const auto& b = word.at("box");
        request.words.push_back({word.at("text"), {b[0], b[1], b[2], b[3]}});
    }
    trtmc::LoadOptions options;
    options.runtime_root = argv[2];
    const auto result = trtmc::Model::load(argv[1], options)
                            .task<trtmc::DocumentWordsToTokenFeatures>()
                            .run(request);
    std::ofstream output(input.at("output_path").get<std::string>(), std::ios::binary);
    output.write(reinterpret_cast<const char*>(result.features().values.data()),
                 result.features().values.size() * sizeof(float));
    nlohmann::json tokens = nlohmann::json::array();
    for (const auto& token : result.tokens())
        tokens.push_back({token.token_id, token.word_index, token.token_index});
    std::cout << nlohmann::json{{"rows", result.features().rows},
                                {"columns", result.features().columns},
                                {"source_words", result.source_word_count()},
                                {"tokens", tokens}}
                     .dump()
              << '\n';
}
