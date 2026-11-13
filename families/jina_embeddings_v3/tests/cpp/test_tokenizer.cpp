/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/jina_embeddings_v3/runtime/tokenizer.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream file(argv[1]);
    const std::string data((std::istreambuf_iterator<char>(file)), {});
    const auto tokenizer = trtmc::CreateUnigramTokenizer(data.data(), data.size(), true);
    std::cout << nlohmann::json(tokenizer->encode(argv[2])).dump() << '\n';
}
