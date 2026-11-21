/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/parakeet_tdt/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

int main(int argc, char** argv) {
    using namespace trtmc::parakeet_tdt;
    std::string source =
        R"({"model":{"type":"BPE","vocab":{"<unk>":0,"<pad>":1,"▁hello":2,"▁world":3,"!":4,"▁你好":5,"Ġ":6},"merges":[]},"added_tokens":[{"id":0,"content":"<unk>","special":true},{"id":1,"content":"<pad>","special":true}],"decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}})";
    nlohmann::json cases = {{{"ids", {6}}, {"text", "Ġ"}},
                            {{"ids", {2, 3, 4}}, {"text", "hello world!"}},
                            {{"ids", {0, 1, 5, 4}}, {"text", "你好!"}},
                            {{"ids", nlohmann::json::array()}, {"text", ""}}};
    if (argc == 3) {
        std::ifstream input(argv[1]);
        std::ostringstream buffer;
        buffer << input.rdbuf();
        source = buffer.str();
        std::ifstream golden(argv[2]);
        golden >> cases;
    }
    auto tokenizer = CreateBpeTokenizer(source.data(), source.size(), false);
    if (!tokenizer)
        return 1;
    for (const auto& item : cases) {
        const auto ids = item.at("ids").get<std::vector<int32_t>>();
        if (tokenizer->decode(ids) != item.at("text").get<std::string>()) {
            std::cerr << "decode mismatch for " << item.at("ids") << '\n';
            return 2;
        }
    }
    std::cout << cases.size() << " tokenizer decode cases passed\n";
}
