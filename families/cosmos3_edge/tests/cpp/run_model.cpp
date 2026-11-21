/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Owning E2E driver for the existing public MC task API, not an installed application.
#include "trtmc/language.hpp"
#include "trtmc/text.hpp"

#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc != 5)
        return 2;
    try {
        const auto model = trtmc::Model::load(argv[1], {argv[2]});
        std::ifstream input(argv[3]);
        nlohmann::json requests, output = nlohmann::json::array();
        input >> requests;
        for (const auto& request : requests) {
            std::deque<std::vector<uint8_t>> pixels;
            std::vector<trtmc::ImageVideoTextPart> parts;
            auto image = [&](const nlohmann::json& spec) {
                std::ifstream file(spec.at("path").get<std::string>(), std::ios::binary);
                if (!file)
                    throw std::runtime_error("Cannot read RGB fixture");
                pixels.emplace_back(std::istreambuf_iterator<char>(file),
                                    std::istreambuf_iterator<char>());
                return trtmc::ImageInput{{pixels.back().data(), pixels.back().size()},
                                         spec.at("height"),
                                         spec.at("width")};
            };
            if (request.contains("image"))
                parts.emplace_back(image(request.at("image")));
            if (request.contains("video")) {
                trtmc::VideoInput video;
                for (const auto& frame : request.at("video").at("frames"))
                    video.frames.push_back(image(frame));
                video.timestamps_seconds =
                    request.at("video").at("timestamps").get<std::vector<double>>();
                parts.emplace_back(std::move(video));
            }
            parts.emplace_back(trtmc::TextPart{request.at("prompt").get<std::string>()});
            trtmc::Config config{{"max_new_tokens", request.value("max_new_tokens", 64)},
                                 {"temperature", request.value("temperature", 0.0)},
                                 {"top_k", request.value("top_k", 1)},
                                 {"top_p", request.value("top_p", 1.0)},
                                 {"seed", request.value("seed", -1)},
                                 {"use_chat_template", request.value("use_chat_template", true)},
                                 {"enable_thinking", request.value("enable_thinking", false)}};
            auto result = [&]() {
                if (!request.contains("image") && !request.contains("video"))
                    return model.task<trtmc::TextContinuation>().run(
                        {request.at("prompt").get<std::string>()}, config);
                if (request.contains("image") && !request.contains("video")) {
                    std::vector<trtmc::ImagesTextPart> image_parts;
                    image_parts.emplace_back(std::get<trtmc::ImageInput>(parts[0]));
                    image_parts.emplace_back(std::get<trtmc::TextPart>(parts.back()));
                    return model.task<trtmc::ImagesTextToText>().run(
                        trtmc::ImagesTextToTextRequest::from_parts(std::move(image_parts)), config);
                }
                if (request.contains("video") && !request.contains("image")) {
                    std::vector<trtmc::VideoTextPart> video_parts;
                    video_parts.emplace_back(std::get<trtmc::VideoInput>(parts[0]));
                    video_parts.emplace_back(std::get<trtmc::TextPart>(parts.back()));
                    return model.task<trtmc::VideoTextToText>().run(
                        trtmc::VideoTextToTextRequest::from_parts(std::move(video_parts)), config);
                }
                return model.task<trtmc::ImageVideoTextToText>().run(
                    trtmc::ImageVideoTextToTextRequest::from_parts(std::move(parts)), config);
            }();
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
