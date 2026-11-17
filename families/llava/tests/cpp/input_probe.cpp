/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llava/runtime/image_preprocess_seam.h"
#include "families/llava/runtime/image_preprocessor.h"
#include "families/llava/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 7)
        return 2;
    std::ifstream tokenizer_file(argv[1]);
    std::string data((std::istreambuf_iterator<char>(tokenizer_file)), {});
    auto tokenizer = trtmc::CreateBpeTokenizer(data.data(), data.size(), true);
    if (!tokenizer)
        throw std::runtime_error("invalid tokenizer");
    const int h = std::stoi(argv[3]), w = std::stoi(argv[4]);
    std::vector<float> pixels(static_cast<std::size_t>(h) * w * 3);
    std::ifstream input(argv[2], std::ios::binary);
    input.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(float));
    if (!input)
        throw std::runtime_error("invalid pixels");
    trtmc::LlavaClipConfig clip;
    clip.input_image_h = clip.input_image_w = 336;
    auto processed = trtmc::preprocess_llava_image(pixels.data(), h, w, clip);
    std::ofstream output(argv[5], std::ios::binary);
    output.write(reinterpret_cast<const char*>(processed.data()), processed.size() * sizeof(float));
    trtmc::LlavaPreprocessConfig config;
    auto prompt = trtmc::llava_format_prompt(argv[6], config, 576);
    std::cout << nlohmann::json({{"input_ids", tokenizer->encode(prompt)}}).dump() << '\n';
}
