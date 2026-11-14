/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Family test helper exposing the native host stages to the E2E oracle:
//   image RUNTIME_JSON RGB_FILE HEIGHT WIDTH OUT_PIXELS OUT_JSON
//     normalized tiles (float32 [frames, 3, tile, tile]) plus rows, cols and
//     the prompt expansion of one raw 8-bit RGB image;
//   tokens TOKENIZER_JSON TEXTS_JSON OUT_JSON
//     token ids of each string of a JSON list;
//   prompt TOKENIZER_JSON REQUEST_JSON OUT_JSON
//     the rendered chat prompt and its token ids for {"image_seq_len": N,
//     "messages": [{"role", "content": [{"type": "text", "text"} |
//     {"type": "image", "rows", "cols"}]}]}.
#include "families/smolvlm/runtime/media.h"
#include "families/smolvlm/runtime/prompt.h"
#include "families/smolvlm/runtime/tokenizer.h"

#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace {

std::string slurp(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error(std::string("cannot read ") + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

int image(char** argv) {
    const auto runtime = nlohmann::json::parse(slurp(argv[2]));
    const auto& p = runtime.at("image_processing");
    trtmc::smolvlm::ImageProcessing processing;
    processing.longest_edge = p.at("longest_edge");
    processing.tile = p.at("tile_size");
    processing.rescale_factor = p.at("rescale_factor");
    processing.mean = p.at("image_mean").get<std::array<float, 3>>();
    processing.std = p.at("image_std").get<std::array<float, 3>>();
    processing.image_seq_len = p.at("image_seq_len");
    trtmc::smolvlm::Rgb8Image rgb;
    rgb.height = std::stoi(argv[4]);
    rgb.width = std::stoi(argv[5]);
    const auto bytes = slurp(argv[3]);
    rgb.pixels.assign(bytes.begin(), bytes.end());
    const auto tiled = trtmc::smolvlm::tile_image(rgb, processing);
    std::ofstream(argv[6], std::ios::binary)
        .write(reinterpret_cast<const char*>(tiled.pixels.data()),
               static_cast<std::streamsize>(tiled.pixels.size() * sizeof(float)));
    std::ofstream(argv[7]) << nlohmann::json{{"rows", tiled.rows},
                                             {"cols", tiled.cols},
                                             {"frames", tiled.frames},
                                             {"prompt", trtmc::smolvlm::image_prompt(
                                                            tiled.rows, tiled.cols,
                                                            processing.image_seq_len)}}
                                  .dump();
    return 0;
}

int tokens(char** argv) {
    const trtmc::smolvlm::Tokenizer tokenizer(slurp(argv[2]));
    auto output = nlohmann::json::array();
    for (const auto& text : nlohmann::json::parse(slurp(argv[3])))
        output.push_back(tokenizer.encode(text.get<std::string>()));
    std::ofstream(argv[4]) << output.dump();
    return 0;
}

int prompt(char** argv) {
    using trtmc::internal::ImagesTextMessage;
    using trtmc::internal::ImagesTextPart;
    using trtmc::internal::MessageRole;
    const trtmc::smolvlm::Tokenizer tokenizer(slurp(argv[2]));
    const auto request = nlohmann::json::parse(slurp(argv[3]));
    std::deque<std::vector<ImagesTextPart>> parts;
    std::deque<std::string> texts;
    std::vector<ImagesTextMessage> messages;
    std::vector<std::pair<int, int>> grids;
    for (const auto& message : request.at("messages")) {
        parts.emplace_back();
        for (const auto& part : message.at("content")) {
            if (part.at("type") == "text") {
                texts.push_back(part.at("text"));
                parts.back().emplace_back(trtmc::internal::TextPartView{texts.back()});
            } else {
                grids.emplace_back(part.at("rows"), part.at("cols"));
                parts.back().emplace_back(trtmc::internal::ImageView{});
            }
        }
        const std::string role = message.at("role");
        if (role != "system" && role != "user" && role != "assistant")
            throw std::invalid_argument("unknown role " + role);
        messages.push_back({role == "system" ? MessageRole::System
                            : role == "user" ? MessageRole::User
                                             : MessageRole::Assistant,
                            {parts.back().data(), parts.back().size()}});
    }
    size_t next = 0;
    const int image_seq_len = request.at("image_seq_len");
    const auto text = trtmc::smolvlm::render_chat(
        {messages.data(), messages.size()}, [&](const trtmc::internal::ImageView&) {
            const auto grid = grids.at(next++);
            return trtmc::smolvlm::image_prompt(grid.first, grid.second, image_seq_len);
        });
    std::ofstream(argv[4])
        << nlohmann::json{{"text", text}, {"ids", tokenizer.encode(text)}}.dump();
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "image" && argc == 8)
            return image(argv);
        if (mode == "tokens" && argc == 5)
            return tokens(argv);
        if (mode == "prompt" && argc == 5)
            return prompt(argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cerr
        << "usage: smolvlm_runtime_probe image RUNTIME_JSON RGB HEIGHT WIDTH OUT_PIXELS OUT_JSON\n"
           "       smolvlm_runtime_probe tokens TOKENIZER_JSON TEXTS_JSON OUT_JSON\n"
           "       smolvlm_runtime_probe prompt TOKENIZER_JSON REQUEST_JSON OUT_JSON\n";
    return 2;
}
