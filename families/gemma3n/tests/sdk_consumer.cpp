/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// C++ SDK consumer of a Gemma 3n bundle: runs images_text_to_text for every
// conversation of a JSON request file and writes the results as JSON.
//
// Usage: test_gemma3n_sdk_cpp <bundle> <runtime_root> <requests.json> <results.json>
//
// A request is {"messages": [{"role": "system"|"user"|"assistant", "parts": [
//   {"text": "..."} | {"image": {"path": <raw HWC RGB>, "height": H, "width": W,
//   "format": "uint8"|"float32"}}]}], "config": {"max_new_tokens": N, ...}}.

#include "trtmc/language.hpp"

#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

trtmc::MessageRole role(const std::string& name) {
    if (name == "system")
        return trtmc::MessageRole::System;
    if (name == "user")
        return trtmc::MessageRole::User;
    if (name == "assistant")
        return trtmc::MessageRole::Assistant;
    throw std::invalid_argument("unknown role " + name);
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot read " + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

trtmc::Config config(const nlohmann::json& values) {
    trtmc::Config result;
    for (auto it = values.begin(); it != values.end(); ++it) {
        if (it.value().is_number_integer())
            result.add(it.key(), it.value().get<std::int64_t>());
        else if (it.value().is_number())
            result.add(it.key(), it.value().get<double>());
        else
            throw std::invalid_argument("config values are numbers: " + it.key());
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0]
                  << " <bundle> <runtime_root> <requests.json> <results.json>\n";
        return 2;
    }
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        const auto model = trtmc::Model::load(argv[1], options);
        if (!model.supports<trtmc::ImagesTextToText>())
            throw std::runtime_error("bundle does not implement images_text_to_text");
        const auto task = model.task<trtmc::ImagesTextToText>();
        nlohmann::json requests, results = nlohmann::json::array();
        std::ifstream(argv[3]) >> requests;
        for (const auto& item : requests) {
            std::deque<std::vector<char>> pixels;
            trtmc::ImagesTextToTextRequest request;
            for (const auto& message : item.at("messages")) {
                trtmc::ImagesTextMessage converted{role(message.at("role")), {}};
                for (const auto& part : message.at("parts")) {
                    if (part.contains("text")) {
                        converted.parts.emplace_back(
                            trtmc::TextPart{part.at("text").get<std::string>()});
                        continue;
                    }
                    const auto& image = part.at("image");
                    pixels.push_back(read_file(image.at("path")));
                    const auto height = image.at("height").get<std::uint32_t>();
                    const auto width = image.at("width").get<std::uint32_t>();
                    const auto& bytes = pixels.back();
                    if (image.value("format", "uint8") == "float32")
                        converted.parts.emplace_back(trtmc::ImageInput{
                            trtmc::Span<const float>{reinterpret_cast<const float*>(bytes.data()),
                                                     bytes.size() / sizeof(float)},
                            height, width});
                    else
                        converted.parts.emplace_back(trtmc::ImageInput{
                            trtmc::Span<const std::uint8_t>{
                                reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()},
                            height, width});
                }
                request.messages.push_back(std::move(converted));
            }
            const auto result =
                task.run(request, config(item.value("config", nlohmann::json::object())));
            const auto ids = result.token_ids();
            results.push_back({{"text", std::string(result.text())},
                               {"token_ids", std::vector<std::int32_t>(ids.begin(), ids.end())},
                               {"prefill_ms", result.prefill_ms()},
                               {"decode_ms", result.decode_ms()}});
            std::cout << results.back().dump() << std::endl;
        }
        std::ofstream(argv[4]) << results.dump(2);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
