/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "chat.h"
#include "media_geometry.h"
#include "tokenizer.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
int main(int argc, char** argv) {
    using namespace trtmc::nemotron_omni;
    using Json = nlohmann::json;
    try {
        if (argc != 2)
            throw std::runtime_error("Expected tokenizer filename");
        std::ifstream file(argv[1]);
        const std::string serialized{std::istreambuf_iterator<char>(file), {}};
        Tokenizer tokenizer(serialized);
        Json input;
        std::cin >> input;
        Json output = Json::array();
        for (const auto& item : input) {
            const std::string mode = item.at("mode");
            if (mode == "chat") {
                std::vector<ChatMessage> messages;
                for (const auto& m : item.at("messages"))
                    messages.push_back({m.at("role"), m.at("content")});
                const auto text = render_chat(messages, item.at("thinking"));
                output.push_back({{"text", text}, {"ids", tokenizer.encode(text)}});
            } else if (mode == "text") {
                const auto ids = tokenizer.encode(item.at("text").get<std::string>());
                output.push_back({{"ids", ids}, {"text", tokenizer.decode(ids)}});
            } else if (mode == "video") {
                const auto grid = video_grid(item.at("height"), item.at("width"), 16, 1024, true);
                output.push_back({grid.rows, grid.columns});
            } else if (mode == "images") {
                std::vector<Grid> shapes;
                for (const auto& shape : item.at("shapes"))
                    shapes.push_back({shape[0], shape[1]});
                Json grids = Json::array();
                for (const auto& grid : image_grids(shapes, 16, 1024, 13312, item.at("tokens")))
                    grids.push_back({grid.rows, grid.columns});
                output.push_back(grids);
            } else
                throw std::runtime_error("Unknown probe mode");
        }
        std::cout << output << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
