/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_vl/runtime/image_processor.h"
#include "families/lfm2_vl/runtime/pipeline.h"
#include "trtmc/language.hpp"

#include <deque>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::ordered_json;
int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument("usage: lfm2_vl_probe BUNDLE RUNTIME_ROOT CASES.json");
        json cases;
        std::ifstream(argv[3]) >> cases;
        std::optional<trtmc::Model> owner;
        std::optional<trtmc::ImagesTextToText> task;
        std::optional<trtmc::TextConversation> conversation;
        if (std::string(argv[1]) != "-") {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            owner.emplace(trtmc::Model::load(argv[1], options));
            if (!owner->supports<trtmc::ImagesTextToText>())
                throw std::runtime_error("bundle does not implement vision generation");
            task.emplace(owner->task<trtmc::ImagesTextToText>());
            conversation.emplace(owner->task<trtmc::TextConversation>());
            for (const auto& field : conversation->config_fields())
                if (field.name == "min_image_tokens" || field.name == "max_image_tokens" ||
                    field.name == "max_tiles" || field.name == "do_image_splitting")
                    throw std::runtime_error("text Task leaked image configuration");
        }
        json results = json::array();
        for (const auto& item : cases) {
            trtmc::Lfm2VlRequest request;
            request.prompt = item.at("prompt").get<std::string>();
            request.generation.max_new_tokens = item.value("max_new_tokens", 32);
            request.generation.use_chat_template = item.value("chat", true);
            request.generation.top_k = item.value("top_k", 1);
            request.generation.temperature = item.value("temperature", 1.0F);
            request.generation.min_p = item.value("min_p", 0.0F);
            request.generation.repetition_penalty = item.value("repetition_penalty", 1.0F);
            request.generation.seed = item.value("seed", 42);
            request.min_image_tokens = item.value("min_image_tokens", -1);
            request.max_image_tokens = item.value("max_image_tokens", -1);
            request.max_tiles = item.value("max_tiles", -1);
            request.split_images = item.value("split_images", -1);
            json metadata = json::array();
            int image_index = 0;
            for (const auto& media : item.value("images", json::array())) {
                trtmc::ImageResult image;
                image.height = media.at("height");
                image.width = media.at("width");
                image.pixels.resize(static_cast<std::size_t>(image.height) * image.width * 3);
                std::ifstream input(media.at("path").get<std::string>(), std::ios::binary);
                input.read(reinterpret_cast<char*>(image.pixels.data()),
                           image.pixels.size() * sizeof(float));
                if (!input)
                    throw std::runtime_error("invalid input image file");
                if (item.contains("dump_prefix")) {
                    trtmc::lfm2_vl::ImageOptions options;
                    if (request.min_image_tokens >= 0)
                        options.min_tokens = request.min_image_tokens;
                    if (request.max_image_tokens >= 0)
                        options.max_tokens = request.max_image_tokens;
                    if (request.max_tiles >= 0)
                        options.max_tiles = request.max_tiles;
                    if (request.split_images >= 0)
                        options.split = request.split_images != 0;
                    auto processed = trtmc::lfm2_vl::process_image(image, options);
                    json tiles = json::array();
                    int index = 0;
                    for (const auto& tile : processed.tiles) {
                        const std::string path = item.at("dump_prefix").get<std::string>() + "-" +
                                                 std::to_string(image_index) + "-" +
                                                 std::to_string(index++) + ".bin";
                        std::ofstream out(path, std::ios::binary);
                        out.write(reinterpret_cast<const char*>(tile.pixels.data()),
                                  tile.pixels.size() * sizeof(float));
                        tiles.push_back(
                            {{"path", path}, {"shape", {tile.patch_rows, tile.patch_cols}}});
                    }
                    metadata.push_back(
                        {{"tiles", tiles}, {"placeholders", processed.placeholders}});
                }
                request.images.push_back(std::move(image));
                ++image_index;
            }
            json result = {{"name", item.at("name")}, {"preprocessing", metadata}};
            if (task) {
                namespace api = trtmc;
                std::vector<std::vector<std::uint8_t>> pixel_bytes;
                std::vector<std::vector<float>> pixel_floats;
                for (const auto& image : request.images) {
                    std::vector<std::uint8_t> bytes;
                    std::vector<float> floats;
                    for (float pixel : image.pixels)
                        bytes.push_back(static_cast<std::uint8_t>(pixel));
                    for (float pixel : image.pixels)
                        floats.push_back(pixel / 255.0F);
                    pixel_bytes.push_back(std::move(bytes));
                    pixel_floats.push_back(std::move(floats));
                }
                std::vector<std::pair<api::MessageRole, std::string>> messages;
                auto role_from_name = [](const std::string& name) {
                    return name == "system"      ? api::MessageRole::System
                           : name == "assistant" ? api::MessageRole::Assistant
                           : name == "tool"      ? api::MessageRole::Tool
                                                 : api::MessageRole::User;
                };
                if (item.contains("semantic_messages"))
                    for (const auto& message : item.at("semantic_messages"))
                        messages.push_back(
                            {role_from_name(message.at("role")), message.value("content", "")});
                else if (request.generation.use_chat_template)
                    messages.push_back({api::MessageRole::User, request.prompt});
                else {
                    std::size_t cursor = 0;
                    while (true) {
                        const auto begin = request.prompt.find("<|im_start|>", cursor);
                        if (begin == std::string::npos)
                            break;
                        const auto newline = request.prompt.find('\n', begin);
                        const auto end = request.prompt.find("<|im_end|>", newline);
                        if (end == std::string::npos)
                            break;
                        const auto name = request.prompt.substr(begin + 12, newline - begin - 12);
                        messages.push_back({role_from_name(name),
                                            request.prompt.substr(newline + 1, end - newline - 1)});
                        cursor = end + 10;
                    }
                    if (messages.empty())
                        throw std::invalid_argument("raw probe input must be rendered ChatML");
                }
                std::deque<std::string> texts;
                auto keep_text = [&](std::string text) -> std::string {
                    texts.push_back(std::move(text));
                    return texts.back();
                };
                std::vector<api::ToolDefinition> tools;
                for (const auto& tool : item.value("tools", json::array()))
                    tools.push_back({keep_text(tool.at("name")), keep_text(tool.at("description")),
                                     keep_text(tool.at("parameters").dump())});
                std::vector<std::vector<api::ImagesTextPart>> part_storage(messages.size());
                std::size_t media_index = 0;
                for (std::size_t m = 0; m < messages.size(); ++m) {
                    if (item.contains("semantic_messages")) {
                        const auto& message = item.at("semantic_messages").at(m);
                        if (message.contains("reasoning"))
                            part_storage[m].push_back(
                                api::ReasoningPart{keep_text(message.at("reasoning"))});
                        if (message.contains("tool_call")) {
                            const auto& call = message.at("tool_call");
                            part_storage[m].push_back(api::ToolCall{
                                keep_text(call.at("call_id")), keep_text(call.at("name")),
                                keep_text(call.at("arguments").dump()),
                                api::ToolCallState::Complete});
                            continue;
                        }
                        if (message.contains("tool_result")) {
                            part_storage[m].push_back(
                                api::ToolResult{keep_text(message.at("call_id")),
                                                keep_text(message.at("tool_result")), false});
                            continue;
                        }
                    }
                    const auto& text = messages[m].second;
                    std::size_t cursor = 0;
                    while (true) {
                        const auto marker = text.find("<image>", cursor);
                        texts.push_back(text.substr(
                            cursor, marker == std::string::npos ? marker : marker - cursor));
                        part_storage[m].push_back(api::TextPart{texts.back()});
                        if (marker == std::string::npos)
                            break;
                        const auto& image = request.images.at(media_index);
                        if (item.value("image_format", "uint8") == "float32") {
                            const auto& pixels = pixel_floats.at(media_index);
                            part_storage[m].push_back(api::ImageInput{
                                api::Span<const float>{pixels.data(), pixels.size()},
                                static_cast<std::uint32_t>(image.height),
                                static_cast<std::uint32_t>(image.width)});
                        } else {
                            const auto& bytes = pixel_bytes.at(media_index);
                            part_storage[m].push_back(api::ImageInput{
                                api::Span<const std::uint8_t>{bytes.data(), bytes.size()},
                                static_cast<std::uint32_t>(image.height),
                                static_cast<std::uint32_t>(image.width)});
                        }
                        ++media_index;
                        cursor = marker + 7;
                    }
                }
                std::vector<api::ImagesTextMessage> views;
                for (std::size_t m = 0; m < messages.size(); ++m)
                    views.push_back({messages[m].first, std::move(part_storage[m])});
                std::vector<api::ConfigEntry> config{
                    {"max_new_tokens", std::int64_t{request.generation.max_new_tokens}},
                    {"top_k", std::int64_t{request.generation.top_k}},
                    {"temperature", static_cast<double>(request.generation.temperature)},
                    {"min_p", static_cast<double>(request.generation.min_p)},
                    {"repetition_penalty",
                     static_cast<double>(request.generation.repetition_penalty)},
                    {"seed", std::int64_t{request.generation.seed}}};
                if (request.min_image_tokens >= 0)
                    config.push_back({"min_image_tokens", std::int64_t{request.min_image_tokens}});
                if (request.max_image_tokens >= 0)
                    config.push_back({"max_image_tokens", std::int64_t{request.max_image_tokens}});
                if (request.max_tiles >= 0)
                    config.push_back({"max_tiles", std::int64_t{request.max_tiles}});
                if (request.split_images >= 0)
                    config.push_back({"do_image_splitting", request.split_images != 0});
                if (item.contains("keep_past_thinking"))
                    config.push_back(
                        {"keep_past_thinking", item.at("keep_past_thinking").get<bool>()});
                if (request.images.empty()) {
                    api::TextConversationRequest input;
                    input.tools = std::move(tools);
                    for (const auto& message : views) {
                        api::ConversationMessage target;
                        target.role = message.role;
                        for (const auto& part : message.parts)
                            std::visit(
                                [&](const auto& value) {
                                    using T = std::decay_t<decltype(value)>;
                                    if constexpr (!std::is_same_v<T, api::ImageInput>)
                                        target.parts.emplace_back(value);
                                },
                                part);
                        input.messages.push_back(std::move(target));
                    }
                    auto output = conversation->run(input, api::Config{std::move(config)});
                    const auto tokens = output.result().token_ids();
                    result["tokens"] = std::vector<std::int32_t>(tokens.begin(), tokens.end());
                    std::string text;
                    result["calls"] = json::array();
                    for (const auto& part : output.message().parts) {
                        if (const auto* value = std::get_if<api::TextPart>(&part))
                            text += value->text;
                        else if (const auto* call = std::get_if<api::ToolCall>(&part))
                            result["calls"].push_back(
                                {{"name", call->name},
                                 {"arguments", json::parse(call->arguments_json)},
                                 {"state", static_cast<int>(call->state)},
                                 {"call_id", call->call_id}});
                        else
                            throw std::runtime_error(
                                "unexpected reasoning output in nonreasoning case");
                    }
                    result["text"] = text;
                } else {
                    auto output = task->run({std::move(views), std::move(tools)},
                                            api::Config{std::move(config)});
                    const auto tokens = output.token_ids();
                    result["tokens"] = std::vector<std::int32_t>(tokens.begin(), tokens.end());
                    result["text"] = output.text();
                }
            }
            results.push_back(result);
            std::cout << result.dump() << std::endl;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
