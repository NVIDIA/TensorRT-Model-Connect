/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_vl_moe/runtime/bpe_tokenizer.cpp"

#include <iostream>

int main() {
    try {
        nlohmann::json request;
        std::cin >> request;
        const auto document = request.at("tokenizer").get<std::string>();
        auto tokenizer = trtmc::CreateBpeTokenizer(document.data(), document.size());
        if (!tokenizer)
            throw std::runtime_error("tokenizer creation failed");
        nlohmann::json results = nlohmann::json::array();
        for (const auto& text : request.at("texts")) {
            const auto value = text.get<std::string>();
            results.push_back({
                {"pieces", trtmc::pretok::pre_tokenize(value, trtmc::pretok::Variant::kQwen3)},
                {"ids", tokenizer->encode(value)},
            });
        }
        std::cout << results.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
