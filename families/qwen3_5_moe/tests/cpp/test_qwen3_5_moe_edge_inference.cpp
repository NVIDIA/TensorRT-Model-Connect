/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/bundle.h"
#include "trtmc/internal/language.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace {
/// Apply only explicitly named public configuration fields; unknown test inputs are errors.
trtmc::TextGenerationConfig generation_config(const nlohmann::json& values) {
    trtmc::TextGenerationConfig result;
    if (!values.is_object())
        throw std::invalid_argument("config must be an object");
    for (const auto& [key, value] : values.items()) {
        if (key == "max_new_tokens")
            result.max_new_tokens = value.get<int32_t>();
        else if (key == "temperature")
            result.temperature = value.get<float>();
        else if (key == "top_k")
            result.top_k = value.get<int32_t>();
        else if (key == "top_p")
            result.top_p = value.get<float>();
        else if (key == "min_p")
            result.min_p = value.get<float>();
        else if (key == "seed")
            result.seed = value.get<int32_t>();
        else if (key == "eos_token_id")
            result.eos_token_id = value.get<int32_t>();
        else if (key == "use_chat_template")
            result.use_chat_template = value.get<bool>();
        else if (key == "enable_thinking")
            result.enable_thinking = value.get<bool>();
        else if (key == "text_generation_mode")
            result.text_generation_mode = value.get<std::string>();
        else if (key == "stop_on_boxed_answer")
            result.stop_on_boxed_answer = value.get<bool>();
        else if (key == "lora_adapter_id")
            result.lora_adapter_id = value.get<std::string>();
        else if (key == "repetition_penalty")
            result.repetition_penalty = value.get<float>();
        else if (key == "block_length")
            result.block_length = value.get<int32_t>();
        else if (key == "confidence_threshold")
            result.confidence_threshold = value.get<float>();
        else if (key == "forced_bos_token_id")
            result.forced_bos_token_id = value.get<int32_t>();
        else if (key == "source_language_token_id")
            result.source_language_token_id = value.get<int32_t>();
        else
            throw std::invalid_argument("Unknown public generation field: " + key);
    }
    return result;
}

/// Parse required qualification inputs without an environment or runtime-root fallback.
std::map<std::string, std::string> arguments(int argc, char** argv) {
    std::map<std::string, std::string> result;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc || !result.emplace(argv[i], argv[i + 1]).second)
            throw std::invalid_argument("Expected unique --option VALUE pairs");
    }
    for (const auto* name : {"--bundle", "--runtime-root", "--requests", "--output"})
        if (!result.count(name))
            throw std::invalid_argument(std::string("Missing ") + name);
    if (result.size() != 4)
        throw std::invalid_argument("Unknown qualification option");
    return result;
}
} // namespace

/// Exercise the public load/generate path once, retaining one task across the complete case list.
int main(int argc, char** argv) {
    try {
        const auto args = arguments(argc, argv);
        const trtmc::BundleReader bundle(args.at("--bundle"));
        if (bundle.info().family != "qwen3_5_moe" || !bundle.find_section("edge_llm.json"))
            throw std::runtime_error("Expected a qwen3_5_moe Edge bundle, not native fallback");
        nlohmann::json report{{"backend", "native"}, {"results", nlohmann::json::array()}};
        if (bundle.find_section("edge_llm.json")) {
            const auto bytes = bundle.read_section("edge_llm.json");
            const auto marker = nlohmann::json::parse(bytes.begin(), bytes.end());
            report["backend"] = "edge_llm";
            report["edge_revision"] = marker.at("edge_revision");
        }
        std::ifstream request_file(args.at("--requests"));
        const std::vector<char> request_bytes((std::istreambuf_iterator<char>(request_file)), {});
        const auto requests = nlohmann::json::parse(request_bytes.begin(), request_bytes.end());
        if (!requests.is_array() || requests.empty())
            throw std::invalid_argument("Requests must be a nonempty JSON array");
        auto task = trtmc::load_task(args.at("--bundle"), args.at("--runtime-root"));
        auto* generator = dynamic_cast<trtmc::ITextGeneration*>(task.get());
        auto* vision = dynamic_cast<trtmc::IVisionLanguageGeneration*>(task.get());
        const bool is_vision = bundle.info().task == "vision_language_generation";
        if ((is_vision && (!vision || generator)) || (!is_vision && (!generator || vision)) ||
            std::string(task->task()) != bundle.info().task)
            throw std::runtime_error("Bundle must expose exactly its selected task interface");
        report["task"] = task->task();
        if (generator)
            report["default_max_new_tokens"] = generator->default_max_new_tokens();
        bool passed = true;
        const auto started = std::chrono::steady_clock::now();
        std::size_t completed = 0;
        std::map<std::string, nlohmann::json> previous;
        for (const auto& request : requests) {
            nlohmann::json output{{"id", request.at("id")}};
            try {
                const auto config =
                    generation_config(request.value("config", nlohmann::json::object()));
                trtmc::TextResult result;
                if (request.contains("video_frames")) {
                    namespace api = trtmc::internal;
                    auto* video = dynamic_cast<api::IVideoTextToText*>(task.get());
                    if (!video)
                        throw std::invalid_argument("Bundle has no typed video task");
                    std::vector<std::vector<std::uint8_t>> storage;
                    std::vector<api::ImageView> frames;
                    const auto h = request.at("height").get<std::uint32_t>();
                    const auto w = request.at("width").get<std::uint32_t>();
                    for (const auto& path : request.at("video_frames")) {
                        std::ifstream file(path.get<std::string>(), std::ios::binary);
                        storage.emplace_back(std::istreambuf_iterator<char>(file),
                                             std::istreambuf_iterator<char>());
                        const auto& bytes = storage.back();
                        if (h == 0 || w == 0 || bytes.size() != static_cast<std::size_t>(h) * w * 3)
                            throw std::invalid_argument("Invalid RGB video frame fixture");
                        frames.push_back(
                            {bytes.data(), bytes.size(), h, w, 3, api::ImageFormat::UInt8});
                    }
                    const auto timestamps = request.value("timestamps", std::vector<double>{});
                    const auto prompt = request.at("prompt").get<std::string>();
                    std::vector<api::VideoTextPart> parts{
                        api::VideoView{{frames.data(), frames.size()},
                                       {timestamps.data(), timestamps.size()}},
                        api::TextPartView{prompt}};
                    api::VideoTextMessage message{api::MessageRole::User,
                                                  {parts.data(), parts.size()}};
                    const std::vector<api::ConfigEntry> options{
                        {"max_new_tokens", std::int64_t(config.max_new_tokens)},
                        {"top_k", std::int64_t(config.top_k)},
                        {"seed", std::int64_t(config.seed)},
                        {"top_p", double(config.top_p)},
                        {"temperature", double(config.temperature)},
                        {"use_chat_template", config.use_chat_template},
                        {"enable_thinking", config.enable_thinking}};
                    result = video->run({{&message, 1}, {}}, {options.data(), options.size()});
                } else if (request.contains("image_file")) {
                    if (!vision)
                        throw std::invalid_argument("Text task cannot accept images");
                    std::ifstream file(request.at("image_file").get<std::string>(),
                                       std::ios::binary);
                    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)),
                                                           {});
                    const auto h = request.at("height").get<int32_t>();
                    const auto w = request.at("width").get<int32_t>();
                    if (h <= 0 || w <= 0 || bytes.size() != static_cast<std::size_t>(h) * w * 3)
                        throw std::runtime_error("Invalid RGB image fixture");
                    std::vector<float> pixels;
                    for (const auto byte : bytes)
                        pixels.push_back(byte / 255.0F);
                    result = vision->generate(request.at("prompt").get<std::string>(),
                                              pixels.data(), h, w, config);
                } else {
                    result =
                        generator
                            ? generator->generate(request.at("prompt").get<std::string>(), config)
                            : vision->generate(request.at("prompt").get<std::string>(), nullptr, 0,
                                               0, config);
                }
                output["text"] = result.text;
                output["token_ids"] = result.token_ids;
            } catch (const std::exception& error) {
                output["error"] = error.what();
            }
            if (output.contains("error") != request.value("expect_error", false))
                passed = false;
            if (request.contains("equal_to")) {
                const auto earlier = previous.find(request.at("equal_to").get<std::string>());
                if (earlier == previous.end() || output.contains("error") ||
                    earlier->second.value("text", "") != output.value("text", "") ||
                    earlier->second.value("token_ids", nlohmann::json::array()) !=
                        output.value("token_ids", nlohmann::json::array()))
                    passed = false;
            }
            if (!output.contains("error") && output.at("token_ids").empty())
                passed = false;
            previous[request.at("id").get<std::string>()] = output;
            report["results"].push_back(std::move(output));
            ++completed;
            if (completed == 1 || completed % 100 == 0 || completed == requests.size()) {
                const auto elapsed =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                        .count();
                std::cerr << "Completed " << completed << "/" << requests.size()
                          << " requests; elapsed_seconds=" << elapsed << '\n';
            }
        }
        report["passed"] = passed;
        task.reset();
        std::ofstream output(args.at("--output"));
        output << report.dump(2) << '\n';
        output.close();
        if (!output)
            throw std::runtime_error("Cannot write qualification output");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
