/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Encodes every text of a JSON array with a checkpoint's tokenizer.json and
// prints one JSON line per text: {"ids": [...]} or {"error": "..."}. The family
// E2E compares these lines with the Hugging Face tokenizer of the checkpoint.

#include "families/openai_privacy_filter/runtime/o200k_tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " TOKENIZER_JSON TEXTS_JSON\n";
        return 2;
    }
    try {
        std::ifstream tokenizer_file(argv[1], std::ios::binary);
        const std::string tokenizer_json((std::istreambuf_iterator<char>(tokenizer_file)),
                                         std::istreambuf_iterator<char>());
        if (!tokenizer_file && !tokenizer_file.eof())
            throw std::runtime_error("cannot read the tokenizer");
        const auto tokenizer =
            trtmc::openai_privacy_filter::O200kTokenizer::from_json(tokenizer_json);
        std::ifstream texts_file(argv[2], std::ios::binary);
        const auto texts = nlohmann::json::parse(texts_file);
        for (const auto& text : texts) {
            nlohmann::json line;
            try {
                line["ids"] = tokenizer.encode(text.get<std::string>());
            } catch (const std::invalid_argument& error) {
                line["error"] = error.what();
            }
            std::cout << line.dump() << '\n';
        }
        std::cout.flush();
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
