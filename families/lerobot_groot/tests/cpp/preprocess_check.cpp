/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Checks the runtime's preprocessing against LeRobot's own, bit for bit:
//
//   trtmc_lerobot_groot_preprocess_check --tokenizer tokenizer.json --golden tokenizer_golden.json
//       [--image CAMERA.png=PROCESSED.png ...] --size N --crop N
//
// The golden lists instructions with the full prompt ids LeRobot's groot preprocessor produced;
// PROCESSED.png is the camera after LeRobot's evaluation crop and INTER_AREA resize. Prints one
// JSON object and exits non-zero on any mismatch.

#include "families/lerobot_groot/runtime/image.h"
#include "families/lerobot_groot/runtime/prompt.h"
#include "families/lerobot_groot/runtime/tokenizer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<std::uint8_t> read_png(const std::string& path, int& side) {
    int width = 0, height = 0, channels = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> decoded(
        stbi_load(path.c_str(), &width, &height, &channels, 3), stbi_image_free);
    if (!decoded || width != height)
        throw std::runtime_error("cannot decode a square RGB image from " + path);
    side = width;
    return {decoded.get(), decoded.get() + static_cast<std::size_t>(width) * height * 3};
}

} // namespace

int main(int argc, char** argv) {
    using namespace trtmc::lerobot_groot;
    try {
        std::string tokenizer_path, golden_path;
        std::vector<std::pair<std::string, std::string>> images;
        int size = 0, crop = 0;
        for (int index = 1; index < argc; ++index) {
            const std::string flag = argv[index];
            if (index + 1 >= argc)
                throw std::runtime_error("missing value for " + flag);
            const std::string value = argv[++index];
            if (flag == "--tokenizer")
                tokenizer_path = value;
            else if (flag == "--golden")
                golden_path = value;
            else if (flag == "--size")
                size = std::stoi(value);
            else if (flag == "--crop")
                crop = std::stoi(value);
            else if (flag == "--image") {
                const auto split = value.find('=');
                if (split == std::string::npos)
                    throw std::runtime_error("--image needs CAMERA=PROCESSED");
                images.emplace_back(value.substr(0, split), value.substr(split + 1));
            } else
                throw std::runtime_error("unknown flag " + flag);
        }
        if (tokenizer_path.empty() || golden_path.empty() || size <= 0 || crop <= 0)
            throw std::runtime_error("--tokenizer, --golden, --size and --crop are required");

        const auto vocabulary = read_file(tokenizer_path);
        const auto tokenizer = CreateBpeTokenizer(vocabulary.data(), vocabulary.size());
        TemplateTokens tokens;
        tokens.im_start = tokenizer->id_for_token("<|im_start|>");
        tokens.im_end = tokenizer->id_for_token("<|im_end|>");
        tokens.vision_start = tokenizer->id_for_token("<|vision_start|>");
        tokens.vision_end = tokenizer->id_for_token("<|vision_end|>");
        tokens.image_pad = tokenizer->id_for_token("<|image_pad|>");
        tokens.user = tokenizer->encode("user\n");
        tokens.newline = tokenizer->encode("\n");

        const auto golden = nlohmann::json::parse(read_file(golden_path));
        const auto cameras = golden.at("cameras").get<std::int32_t>();
        const auto grid = golden.at("grid").get<std::int32_t>();
        int matched = 0, mismatched = 0;
        nlohmann::json failures = nlohmann::json::array();
        for (const auto& item : golden.at("cases")) {
            const auto text = item.at("text").get<std::string>();
            const auto expected = item.at("prompt_ids").get<std::vector<std::int32_t>>();
            const auto prompt = build_prompt(tokenizer->encode(formalize_instruction(text, true)),
                                             cameras, grid, tokens);
            if (prompt.token_ids == expected) {
                ++matched;
            } else {
                ++mismatched;
                failures.push_back(
                    {{"text", text}, {"expected", expected}, {"got", prompt.token_ids}});
            }
        }

        nlohmann::json image_results = nlohmann::json::array();
        bool images_exact = true;
        for (const auto& [camera_path, processed_path] : images) {
            int camera_side = 0, processed_side = 0;
            const auto camera = read_png(camera_path, camera_side);
            const auto processed = read_png(processed_path, processed_side);
            if (camera_side != size || processed_side != size)
                throw std::runtime_error("image sides do not match --size");
            std::vector<float> ours, theirs;
            append_patches(camera.data(), size, crop, ours);
            append_patches(processed.data(), size, size, theirs);
            const bool exact = ours == theirs;
            images_exact = images_exact && exact;
            image_results.push_back({{"camera", camera_path}, {"exact", exact}});
        }

        const bool ok = mismatched == 0 && images_exact;
        std::cout << nlohmann::json{{"tokenizer_cases", matched + mismatched},
                                    {"tokenizer_mismatches", mismatched},
                                    {"failures", failures},
                                    {"images", image_results},
                                    {"ok", ok}}
                         .dump()
                  << '\n';
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
