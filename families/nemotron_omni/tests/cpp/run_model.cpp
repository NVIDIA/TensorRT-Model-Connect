/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Owning test driver for actual public MC tasks, not a second inference runtime.
#include "trtmc/language.hpp"
#include "trtmc/text.hpp"

#include <chrono>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;
std::vector<char> bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open fixture: " + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
int main(int argc, char** argv) {
    if (argc != 5)
        return 2;
    try {
        trtmc::LoadOptions options{};
        options.runtime_root = argv[2];
        const auto load_start = std::chrono::steady_clock::now();
        std::cerr << "Loading Omni model" << std::endl;
        const auto model = trtmc::Model::load(argv[1], options);
        std::cerr
            << "Omni model loaded in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count()
            << " seconds" << std::endl;
        std::ifstream file(argv[3]);
        Json requests, output = Json::array();
        file >> requests;
        bool failed = false;
        for (const auto& request : requests) {
            Json record{{"id", request.at("id")}};
            const auto request_start = std::chrono::steady_clock::now();
            std::cerr << "Running Omni case " << request.at("id") << std::endl;
            try {
                std::deque<std::vector<char>> pixels;
                auto image = [&](const Json& spec) {
                    pixels.push_back(bytes(spec.at("path")));
                    return trtmc::ImageInput{
                        {reinterpret_cast<const uint8_t*>(pixels.back().data()),
                         pixels.back().size()},
                        spec.at("height"),
                        spec.at("width")};
                };
                trtmc::VideoInput video;
                if (request.contains("video")) {
                    for (const auto& frame : request.at("video").at("frames"))
                        video.frames.push_back(image(frame));
                    video.timestamps_seconds =
                        request.at("video").at("timestamps").get<std::vector<double>>();
                }
                std::vector<float> pcm;
                trtmc::AudioView audio;
                if (request.contains("audio")) {
                    const auto raw = bytes(request.at("audio").at("path"));
                    if (raw.empty() || raw.size() % sizeof(float))
                        throw std::runtime_error("Invalid float PCM fixture");
                    pcm.resize(raw.size() / sizeof(float));
                    std::memcpy(pcm.data(), raw.data(), raw.size());
                    audio = {{pcm.data(), pcm.size()},
                             request.at("audio").at("sample_rate"),
                             request.at("audio").at("channels")};
                }
                const trtmc::TextPart text{request.at("prompt").get<std::string>()};
                trtmc::Config config{
                    {"max_new_tokens", request.value("max_new_tokens", 96)},
                    {"temperature", request.value("temperature", 0.0)},
                    {"top_k", request.value("top_k", 1)},
                    {"top_p", request.value("top_p", 1.0)},
                    {"seed", request.value("seed", -1)},
                    {"use_chat_template", request.value("use_chat_template", true)},
                    {"enable_thinking", request.value("enable_thinking", false)}};
                auto result = [&]() {
                    if (request.contains("audio") && request.contains("video")) {
                        trtmc::AlignedAudioVideoInput aligned{
                            std::move(video), audio, request.value("audio_start_seconds", 0.)};
                        return model.task<trtmc::AudioVideoTextToText>().run(
                            trtmc::AudioVideoTextToTextRequest::from_parts(
                                {std::move(aligned), text}),
                            config);
                    }
                    if (request.contains("audio") && request.contains("image"))
                        return model.task<trtmc::ImageAudioTextToText>().run(
                            trtmc::ImageAudioTextToTextRequest::from_parts(
                                {image(request.at("image")), audio, text}),
                            config);
                    if (request.contains("audio"))
                        return model.task<trtmc::AudioTextToText>().run(
                            trtmc::AudioTextToTextRequest::from_parts({audio, text}), config);
                    if (request.contains("video"))
                        return model.task<trtmc::VideoTextToText>().run(
                            trtmc::VideoTextToTextRequest::from_parts({std::move(video), text}),
                            config);
                    if (request.contains("image"))
                        return model.task<trtmc::ImagesTextToText>().run(
                            trtmc::ImagesTextToTextRequest::from_parts(
                                {image(request.at("image")), text}),
                            config);
                    return model.task<trtmc::TextContinuation>().run({text.text}, config);
                }();
                const auto ids = result.token_ids();
                record.update({{"status", "passed"},
                               {"text", std::string(result.text())},
                               {"token_ids", std::vector<int32_t>(ids.begin(), ids.end())}});
            } catch (const std::exception& error) {
                record.update({{"status", "command_failure"}, {"error", error.what()}});
                failed = true;
            }
            record["elapsed_seconds"] =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - request_start)
                    .count();
            output.push_back(record);
            std::cout << record.dump() << std::endl;
            std::ofstream(argv[4]) << output.dump(2);
        }
        return failed ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
