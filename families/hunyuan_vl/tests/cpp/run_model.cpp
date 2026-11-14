/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Owning E2E driver for the public images-text-to-text task API, not an installed application.
#include "trtmc/language.hpp"

#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr
            << "usage: hunyuan_vl_inference_test <bundle> <runtime> <requests.json> <out.json>\n";
        return 2;
    }
    try {
        const auto model = trtmc::Model::load(argv[1], {argv[2]});
        std::ifstream input(argv[3]);
        nlohmann::json requests, output = nlohmann::json::array();
        input >> requests;
        for (const auto& request : requests) {
            std::deque<std::vector<uint8_t>> pixels;
            trtmc::ImagesTextToTextRequest task_request;
            for (const auto& message : request.at("messages")) {
                const auto role = message.at("role").get<std::string>();
                trtmc::ImagesTextMessage converted{role == "system" ? trtmc::MessageRole::System
                                                   : role == "assistant"
                                                       ? trtmc::MessageRole::Assistant
                                                       : trtmc::MessageRole::User,
                                                   {}};
                for (const auto& part : message.at("parts")) {
                    if (part.contains("text")) {
                        converted.parts.emplace_back(
                            trtmc::TextPart{part.at("text").get<std::string>()});
                        continue;
                    }
                    const auto& image = part.at("image");
                    std::ifstream file(image.at("path").get<std::string>(), std::ios::binary);
                    if (!file)
                        throw std::runtime_error("Cannot read RGB fixture");
                    pixels.emplace_back(std::istreambuf_iterator<char>(file),
                                        std::istreambuf_iterator<char>());
                    converted.parts.emplace_back(
                        trtmc::ImageInput{{pixels.back().data(), pixels.back().size()},
                                          image.at("height").get<uint32_t>(),
                                          image.at("width").get<uint32_t>()});
                }
                task_request.messages.push_back(std::move(converted));
            }
            trtmc::Config config{{"max_new_tokens", request.value("max_new_tokens", 128)},
                                 {"repetition_penalty", request.value("repetition_penalty", 1.0)}};
            const auto result = model.task<trtmc::ImagesTextToText>().run(task_request, config);
            const auto ids = result.token_ids();
            output.push_back({{"text", std::string(result.text())},
                              {"token_ids", std::vector<int32_t>(ids.begin(), ids.end())}});
            std::cout << output.back().dump() << '\n';
            std::ofstream(argv[4]) << output.dump(2);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
