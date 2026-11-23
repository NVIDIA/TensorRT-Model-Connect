/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/language.hpp"
#include "trtmc/text.hpp"

#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: sdk_consumer BUNDLE RUNTIME_ROOT REQUESTS ANSWERS\n";
        return 2;
    }
    try {
        trtmc::LoadOptions options{};
        options.runtime_root = argv[2];
        const auto model = trtmc::Model::load(argv[1], options);
        const auto requests = Json::parse(std::ifstream(argv[3]));
        Json answers = Json::array();
        const trtmc::Config config{
            {"max_new_tokens", 96}, {"temperature", 0.0},        {"top_k", 1},
            {"top_p", 1.0},         {"use_chat_template", true}, {"enable_thinking", false}};
        for (const auto& request : requests) {
            std::deque<std::vector<uint8_t>> pixels;
            auto image = [&](const Json& spec) {
                std::ifstream input(spec.at("path").get<std::string>(), std::ios::binary);
                if (!input)
                    throw std::runtime_error("Missing RGB fixture");
                pixels.emplace_back(std::istreambuf_iterator<char>{input},
                                    std::istreambuf_iterator<char>{});
                const auto h = spec.at("height").get<int>();
                const auto w = spec.at("width").get<int>();
                const auto& rgb = pixels.back();
                if (h <= 0 || w <= 0 || rgb.size() != static_cast<size_t>(h) * w * 3)
                    throw std::runtime_error("Invalid RGB fixture dimensions");
                return trtmc::ImageInput{
                    {rgb.data(), rgb.size()}, static_cast<uint32_t>(h), static_cast<uint32_t>(w)};
            };
            const trtmc::TextPart text{request.at("prompt").get<std::string>()};
            auto result = [&]() {
                if (request.contains("video")) {
                    trtmc::VideoInput video;
                    for (const auto& frame : request.at("video").at("frames"))
                        video.frames.push_back(image(frame));
                    video.timestamps_seconds =
                        request.at("video").at("timestamps").get<std::vector<double>>();
                    return model.task<trtmc::VideoTextToText>().run(
                        trtmc::VideoTextToTextRequest::from_parts({std::move(video), text}),
                        config);
                }
                if (request.contains("image"))
                    return model.task<trtmc::ImagesTextToText>().run(
                        trtmc::ImagesTextToTextRequest::from_parts(
                            {image(request.at("image")), text}),
                        config);
                return model.task<trtmc::TextContinuation>().run({text.text}, config);
            }();
            auto ids = result.token_ids();
            answers.push_back({{"id", request.at("id")},
                               {"text", std::string(result.text())},
                               {"token_ids", std::vector<int32_t>(ids.begin(), ids.end())}});
        }
        std::ofstream out(argv[4]);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out << answers.dump(2);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
