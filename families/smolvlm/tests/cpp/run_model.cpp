/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Owning E2E driver for the public images_text_to_text Task, not an installed application.
// Usage: smolvlm_inference_test BUNDLE RUNTIME_ROOT REQUESTS.json OUTPUT.json
#include "trtmc/language.hpp"
#include "trtmc/text.hpp"

#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>

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

} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr
            << "usage: smolvlm_inference_test BUNDLE RUNTIME_ROOT REQUESTS.json OUTPUT.json\n";
        return 2;
    }
    try {
        const auto model = trtmc::Model::load(argv[1], {argv[2]});
        const auto task = model.task<trtmc::ImagesTextToText>();
        std::ifstream input(argv[3]);
        nlohmann::json requests, output = nlohmann::json::array();
        input >> requests;
        for (const auto& request : requests) {
            std::deque<std::vector<uint8_t>> pixels;
            trtmc::ImagesTextToTextRequest call;
            for (const auto& message : request.at("messages")) {
                trtmc::ImagesTextMessage converted{role(message.at("role")), {}};
                for (const auto& part : message.at("content")) {
                    if (part.at("type") == "text") {
                        converted.parts.emplace_back(trtmc::TextPart{part.at("text")});
                        continue;
                    }
                    std::ifstream file(part.at("path").get<std::string>(), std::ios::binary);
                    if (!file)
                        throw std::runtime_error("cannot read the RGB fixture");
                    pixels.emplace_back(std::istreambuf_iterator<char>(file),
                                        std::istreambuf_iterator<char>());
                    converted.parts.emplace_back(
                        trtmc::ImageInput{{pixels.back().data(), pixels.back().size()},
                                          part.at("height").get<std::uint32_t>(),
                                          part.at("width").get<std::uint32_t>()});
                }
                call.messages.push_back(std::move(converted));
            }
            trtmc::Config config;
            // Keep the settings alive: items() only references its JSON object.
            const auto settings = request.value("config", nlohmann::json::object());
            for (const auto& [name, value] : settings.items()) {
                if (value.is_number_integer())
                    config.add(name, value.get<std::int64_t>());
                else
                    config.add(name, value.get<double>());
            }
            const auto result = task.run(call, config);
            const auto ids = result.token_ids();
            output.push_back({{"text", std::string(result.text())},
                              {"token_ids", std::vector<std::int32_t>(ids.begin(), ids.end())},
                              {"prefill_ms", result.prefill_ms()},
                              {"decode_ms", result.decode_ms()}});
            std::cout << output.back().dump() << '\n';
        }
        std::ofstream(argv[4]) << output.dump(2);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
