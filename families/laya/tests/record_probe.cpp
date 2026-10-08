/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/laya/runtime/record.h"

#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: laya_record_probe TOKENIZER CONFIG");
        std::ifstream input(argv[1]);
        const std::string data((std::istreambuf_iterator<char>(input)), {});
        const auto tokenizer = trtmc::laya::CreateBpeTokenizer(data.data(), data.size());
        std::ifstream config_file(argv[2]);
        const auto config = trtmc::laya::Json::parse(config_file);
        std::string line;
        while (std::getline(std::cin, line)) {
            const auto record = trtmc::laya::Json::parse(line);
            if (record.is_string()) {
                std::cout << trtmc::laya::Json(tokenizer->encode(record.get<std::string>())).dump()
                          << '\n';
            } else {
                auto result = trtmc::laya::Json::array();
                for (const auto& q : trtmc::laya::encode_record(*tokenizer, record, config))
                    result.push_back({{"id", q.id},
                                      {"type", q.type},
                                      {"options", q.options},
                                      {"tokens", q.tokens},
                                      {"markers", q.markers}});
                std::cout << result.dump() << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
