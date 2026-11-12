/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument(
                "expected bundle, runtime root, reference directory, manifest");
        const std::filesystem::path root(argv[3]);
        const auto reference = nlohmann::json::parse(std::ifstream(root / "reference.json"));
        const auto manifest = nlohmann::json::parse(std::ifstream(argv[4]));
        auto task = trtmc::load_task(argv[1], argv[2]);
        auto* vision = dynamic_cast<trtmc::IVisionLanguageGeneration*>(task.get());
        if (!vision)
            throw std::runtime_error("Molmo2 family does not implement IVisionLanguageGeneration");
        nlohmann::json outputs = nlohmann::json::array();
        // Deliberately reuse the loaded engines and caches across different images,
        // an overlength rejection, and a final repeat of the first request.
        for (std::size_t i = 0; i <= reference.size(); ++i) {
            const std::size_t index = i % reference.size();
            const auto& record = reference.at(index);
            const auto& testcase = manifest.at("testcases").at(index);
            const std::string name = record.at("name");
            const int h = record.at("height"), w = record.at("width");
            std::vector<float> pixels(static_cast<std::size_t>(h) * w * 3);
            std::ifstream input(root / (name + ".f32"), std::ios::binary);
            input.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(float));
            if (!input)
                throw std::runtime_error("could not read image " + name);
            for (auto& value : pixels)
                value /= 255.0F;
            trtmc::TextGenerationConfig config;
            config.max_new_tokens = testcase.at("max_new_tokens");
            if (i == reference.size()) {
                auto invalid = config;
                invalid.max_new_tokens = 1000000;
                bool rejected = false;
                try {
                    vision->generate(testcase.at("prompt"), pixels.data(), h, w, invalid);
                } catch (const std::invalid_argument&) {
                    rejected = true;
                }
                if (!rejected)
                    throw std::runtime_error("overlength request was accepted");
            }
            auto result = vision->generate(testcase.at("prompt"), pixels.data(), h, w, config);
            outputs.push_back({{"name", name},
                               {"text", result.text},
                               {"token_ids", result.token_ids},
                               {"prefill_ms", result.prefill_ms},
                               {"decode_ms", result.decode_ms}});
        }
        std::cout << outputs.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
