/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/clef/runtime/media.h"
#include "families/clef/runtime/media_io.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument("usage: media_probe TOKENIZER PROCESSOR RECORD OUTPUT");
        std::ifstream tf(argv[1]);
        const std::string tokenizer_data((std::istreambuf_iterator<char>(tf)), {});
        auto tokenizer =
            trtmc::CreateBpeTokenizer(tokenizer_data.data(), tokenizer_data.size(), false);
        std::ifstream pf(argv[2]);
        const auto processor = trtmc::clef::Json::parse(pf);
        const auto request = trtmc::clef::read_request(argv[3]);
        const auto document = trtmc::clef::Json::parse(request.document);
        const auto media = trtmc::clef::preprocess_media(*tokenizer, request, document, processor);
        const auto record =
            trtmc::clef::encode_record(*tokenizer, document, 16384, -1, media.tokens);
        const auto positions =
            trtmc::clef::media_positions(record, media, tokenizer->id_for_token("<|image_pad|>"),
                                         tokenizer->id_for_token("<|video_pad|>"));
        const std::filesystem::path root(argv[4]);
        std::filesystem::create_directories(root);
        nlohmann::json grids = nlohmann::json::array();
        for (std::size_t i = 0; i < media.frames.size(); ++i) {
            const auto& frame = media.frames[i];
            std::ofstream out(root / (std::to_string(i) + ".bin"), std::ios::binary);
            out.write(reinterpret_cast<const char*>(frame.patches.data()),
                      frame.patches.size() * sizeof(float));
            grids.push_back({1, frame.grid_height, frame.grid_width});
        }
        std::cout << nlohmann::json({{"media_tokens", media.tokens},
                                     {"input_ids", record.input_ids},
                                     {"positions", positions},
                                     {"grids", grids}})
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
