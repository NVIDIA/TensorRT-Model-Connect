/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runs the native GLM-OCR host steps on caller-supplied inputs so that the host-parity tests
// of tests/test_e2e.py can compare them with the Transformers processor and the tokenizers
// library. It is part of test_glm_ocr_runtime (the CTest target), which runs it when given
// a command:
//   test_glm_ocr_runtime preprocess <runtime.json> <image.rgb8> <height> <width> <out prefix>
//   test_glm_ocr_runtime tokenize <tokenizer.json> <texts.json> <out.json>
//   test_glm_ocr_runtime pretokenize <texts.json> <out.json>
//   test_glm_ocr_runtime prompt <tokenizer.json> <conversations.json> <out.json>

#include "families/glm_ocr/runtime/image.h"
#include "families/glm_ocr/runtime/model.h"
#include "families/glm_ocr/runtime/prompt.h"
#include "families/glm_ocr/runtime/tokenizer.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<char> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_floats(const std::string& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output)
        throw std::runtime_error("cannot write " + path);
}

void write_json(const std::string& path, const nlohmann::json& value) {
    std::ofstream output(path, std::ios::trunc);
    output << value.dump();
    if (!output)
        throw std::runtime_error("cannot write " + path);
}

int preprocess(char** argv) {
    const auto config = trtmc::glm_ocr::parse_runtime_config(read_file(argv[2]));
    const auto bytes = read_file(argv[3]);
    const auto height = static_cast<std::uint32_t>(std::stoul(argv[4]));
    const auto width = static_cast<std::uint32_t>(std::stoul(argv[5]));
    const trtmc::internal::ImageView view{
        bytes.data(), bytes.size(), height, width, 3, trtmc::internal::ImageFormat::UInt8};
    const auto input = trtmc::glm_ocr::preprocess_image(view, config.image);
    const std::string prefix = argv[6];
    write_floats(prefix + ".patches.f32", input.patches);
    write_floats(prefix + ".cos.f32", input.rotary_cos);
    write_floats(prefix + ".sin.f32", input.rotary_sin);
    std::printf("{\"grid_h\": %d, \"grid_w\": %d}\n", input.grid_h, input.grid_w);
    return 0;
}

int tokenize(char** argv) {
    const auto json = read_file(argv[2]);
    const trtmc::glm_ocr::Tokenizer tokenizer(std::string_view(json.data(), json.size()));
    const auto texts = nlohmann::json::parse(read_file(argv[3]));
    nlohmann::json results = nlohmann::json::array();
    for (const auto& text : texts) {
        const auto ids = tokenizer.encode(text.get<std::string>());
        const trtmc::Span<const std::int32_t> span(ids.data(), ids.size());
        results.push_back({{"ids", ids},
                           {"decoded", tokenizer.decode(span, true)},
                           {"decoded_with_special", tokenizer.decode(span, false)}});
    }
    write_json(argv[4], results);
    return 0;
}

// Byte lengths of the pre-tokenizer pieces of each text.
int pretokenize(char** argv) {
    const auto texts = nlohmann::json::parse(read_file(argv[2]));
    nlohmann::json results = nlohmann::json::array();
    for (const auto& text : texts) {
        const auto value = text.get<std::string>();
        std::vector<std::size_t> lengths;
        for (const auto piece : trtmc::glm_ocr::pre_tokenize(value))
            lengths.push_back(piece.size());
        results.push_back(std::move(lengths));
    }
    write_json(argv[3], results);
    return 0;
}

trtmc::internal::MessageRole role_of(const std::string& name) {
    using trtmc::internal::MessageRole;
    if (name == "system")
        return MessageRole::System;
    if (name == "user")
        return MessageRole::User;
    if (name == "assistant")
        return MessageRole::Assistant;
    throw std::runtime_error("unknown role " + name);
}

int prompt(char** argv) {
    const auto json = read_file(argv[2]);
    const trtmc::glm_ocr::Tokenizer tokenizer(std::string_view(json.data(), json.size()));
    const std::int32_t image_token = tokenizer.added_token_id("<|image|>");
    const auto conversations = nlohmann::json::parse(read_file(argv[3]));
    const float pixel[3] = {0, 0, 0};
    const trtmc::internal::ImageView image{
        pixel, sizeof(pixel), 1, 1, 3, trtmc::internal::ImageFormat::Float32};
    nlohmann::json results = nlohmann::json::array();
    for (const auto& conversation : conversations) {
        std::vector<std::string> texts;
        std::vector<std::vector<trtmc::internal::ImagesTextPart>> parts;
        std::vector<trtmc::glm_ocr::ImageGrid> grids;
        std::vector<trtmc::internal::MessageRole> roles;
        for (const auto& message : conversation.at("messages")) {
            roles.push_back(role_of(message.at("role").get<std::string>()));
            for (const auto& part : message.at("parts"))
                if (part.contains("text"))
                    texts.push_back(part.at("text").get<std::string>());
        }
        std::size_t text_index = 0;
        for (const auto& message : conversation.at("messages")) {
            parts.emplace_back();
            for (const auto& part : message.at("parts")) {
                if (part.contains("text")) {
                    parts.back().push_back(trtmc::internal::TextPartView{texts[text_index++]});
                } else {
                    const auto grid = part.at("image").get<std::vector<int>>();
                    grids.push_back({grid.at(0), grid.at(1)});
                    parts.back().push_back(image);
                }
            }
        }
        std::vector<trtmc::internal::ImagesTextMessage> messages;
        for (std::size_t i = 0; i < parts.size(); ++i)
            messages.push_back({roles[i], {parts[i].data(), parts[i].size()}});
        const trtmc::internal::ImagesTextToTextRequest request{{messages.data(), messages.size()},
                                                               {}};
        const auto rendered = trtmc::glm_ocr::render_prompt(request);
        const auto layout =
            trtmc::glm_ocr::layout_prompt(tokenizer.encode(rendered.text), image_token, grids, 2);
        results.push_back({{"text", rendered.text},
                           {"ids", layout.ids},
                           {"positions", layout.positions},
                           {"next_position", layout.next_position}});
    }
    write_json(argv[4], results);
    return 0;
}

} // namespace

namespace trtmc::glm_ocr::testing {

int run_host_probe(int argc, char** argv) {
    try {
        const std::string command = argv[1];
        if (command == "preprocess" && argc == 7)
            return preprocess(argv);
        if (command == "tokenize" && argc == 5)
            return tokenize(argv);
        if (command == "pretokenize" && argc == 4)
            return pretokenize(argv);
        if (command == "prompt" && argc == 5)
            return prompt(argv);
        std::fprintf(stderr, "usage: see the comment at the top of host_probe.cpp\n");
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "glm_ocr host probe: %s\n", error.what());
        return 1;
    }
}

} // namespace trtmc::glm_ocr::testing
