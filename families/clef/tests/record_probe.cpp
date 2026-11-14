/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/clef/runtime/record.h"

#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char** argv) {
    try {
        if (argc < 3)
            throw std::invalid_argument(
                "usage: record_probe TOKENIZER RECORD [MAX_LENGTH] [MAX_STATE]");
        std::ifstream tokenizer_file(argv[1]);
        const std::string data((std::istreambuf_iterator<char>(tokenizer_file)), {});
        auto tokenizer = trtmc::CreateBpeTokenizer(data.data(), data.size(), false);
        std::ifstream record_file(argv[2]);
        const auto document = trtmc::clef::Json::parse(record_file);
        const auto result =
            trtmc::clef::encode_record(*tokenizer, document, argc > 3 ? std::stoi(argv[3]) : 16384,
                                       argc > 4 ? std::stoi(argv[4]) : -1);
        nlohmann::json out = {{"input_ids", result.input_ids},
                              {"questions", nlohmann::json::array()}};
        for (const auto& q : result.questions) {
            out["questions"].push_back({{"question_id", q.id},
                                        {"question_type", q.type},
                                        {"question_span", q.span},
                                        {"option_spans", q.option_spans},
                                        {"option_ids", q.option_ids}});
        }
        std::cout << out.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
