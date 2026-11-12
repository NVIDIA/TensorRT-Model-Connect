/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/molmo2/runtime/preprocess.h"
#include "families/molmo2/runtime/tokenizer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class T>
void write(const std::string& path, const std::vector<T>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    require(static_cast<bool>(output), "could not write preprocessing fixture");
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 7) {
            int h = std::stoi(argv[1]), w = std::stoi(argv[2]);
            std::vector<float> pixels(static_cast<std::size_t>(h) * w * 3);
            std::ifstream input(argv[3], std::ios::binary);
            input.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(float));
            require(static_cast<bool>(input), "could not read image fixture");
            for (auto& value : pixels)
                value /= 255.0F;
            auto processed = trtmc::molmo2::preprocess(pixels.data(), h, w);
            write(std::string(argv[4]) + ".pixels", processed.pixels);
            write(std::string(argv[4]) + ".pooling", processed.pooling);
            std::ifstream file(argv[5]);
            std::string tokenizer((std::istreambuf_iterator<char>(file)), {});
            auto bpe = trtmc::molmo2::CreateBpeTokenizer(tokenizer.data(), tokenizer.size(), false);
            const std::string text = trtmc::molmo2::image_prompt(processed) + "<|im_start|>user\n" +
                                     argv[6] + "<|im_end|>\n<|im_start|>assistant\n";
            auto ids = bpe->encode(text);
            ids.insert(ids.begin(), 151645);
            nlohmann::json result{{"crops", processed.crops},
                                  {"height", processed.pooled_height},
                                  {"width", processed.pooled_width},
                                  {"ids", ids}};
            std::ofstream(std::string(argv[4]) + ".json") << result;
            return 0;
        }
        const auto mask = trtmc::molmo2::attention_mask({151645, 151940, 151938, 151937, 100}, 0);
        require(mask.size() == 30, "unexpected mask dimensions");
        require(mask[1 * 6 + 3] == 0 && mask[3 * 6 + 2] == 0,
                "image attention must be bidirectional");
        require(mask[0 * 6 + 2] < -1e20F && mask[1 * 6 + 5] < -1e20F, "text must remain causal");
        for (int q = 0; q < 5; ++q)
            require(mask[q * 6] < -1e20F, "sentinel key must be masked");
        const auto decode = trtmc::molmo2::attention_mask({100}, 8);
        require(decode.size() == 10 && decode.front() < -1e20F && decode.back() == 0,
                "decode mask must retain valid cache history");
        std::vector<float> red(32 * 32 * 3, 0);
        for (int i = 0; i < 32 * 32; ++i)
            red[i * 3] = 1;
        const auto image = trtmc::molmo2::preprocess(red.data(), 32, 32);
        require(image.crops == 2 && image.pooling.size() == 392 * 4,
                "small image must include both overview and high resolution crop");
        require(image.pixels[0] == 1 && image.pixels[1] == -1 && image.pixels[2] == -1,
                "RGB normalization changed");
        require(std::count(image.pooling.begin(), image.pooling.end(), -1) == 110,
                "odd patch grids must preserve pooling padding");
        std::cout << "Molmo2 preprocessing and attention contracts passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
