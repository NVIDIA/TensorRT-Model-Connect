/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/layoutlmv3/runtime/document_input.h"
#include "families/layoutlmv3/runtime/image_preprocess.h"

#include <fstream>
#include <iostream>
int main() {
    nlohmann::json input;
    std::cin >> input;
    const uint32_t h = input.at("height"), w = input.at("width");
    std::vector<float> pixels(static_cast<size_t>(h) * w * 3);
    std::ifstream image(input.at("image_path").get<std::string>(), std::ios::binary);
    image.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(float));
    if (!image)
        return 2;
    std::ifstream file(input.at("tokenizer_path").get<std::string>(), std::ios::binary);
    const std::string data((std::istreambuf_iterator<char>(file)), {});
    auto tokenizer = trtmc::CreateBpeTokenizer(data.data(), data.size(), false);
    std::vector<std::string> texts;
    for (const auto& word : input.at("words"))
        texts.push_back(word.at("text"));
    std::vector<trtmc::internal::DocumentWordView> words;
    for (size_t i = 0; i < texts.size(); ++i) {
        const auto& b = input.at("words")[i].at("box");
        words.push_back({texts[i], {b[0], b[1], b[2], b[3]}});
    }
    const trtmc::internal::ImageView view{pixels.data(),
                                          pixels.size() * sizeof(float),
                                          h,
                                          w,
                                          3,
                                          trtmc::internal::ImageFormat::Float32};
    const auto result = trtmc::layoutlmv3::encode_document(
        *tokenizer, {view, {words.data(), words.size()}}, input.at("config"));
    const auto processed = trtmc::layoutlmv3::image_pixels(view);
    const auto write = [&](const char* key, const auto& values) {
        std::ofstream out(input.at(key).template get<std::string>(), std::ios::binary);
        out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(values[0]));
        out.close();
        if (!out)
            std::cerr << "Could not write LayoutLMv3 preprocessing output: " << input.at(key)
                      << '\n';
        return static_cast<bool>(out);
    };
    if (!write("pixels_path", processed) || !write("relative_x_path", result.relative_x) ||
        !write("relative_y_path", result.relative_y))
        return 2;
    std::vector<int64_t> mapping;
    for (const auto& token : result.mapping)
        mapping.push_back(token.word_index);
    std::cout << nlohmann::json{{"input_ids", result.ids},
                                {"attention_mask", result.valid},
                                {"position_ids", result.positions},
                                {"bbox", result.boxes},
                                {"word_indices", mapping}}
                     .dump()
              << '\n';
}
