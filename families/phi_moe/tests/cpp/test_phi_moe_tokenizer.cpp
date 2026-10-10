/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/phi_moe/runtime/tokenizer.h"

#include <iostream>

int main() {
    const std::string source = R"json({
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "Ġ": 2, "Ċ": 3, "ĉ": 4}, "merges": []},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
        "added_tokens": [
            {"id": 10, "content": "<|user|>", "special": true, "lstrip": true, "rstrip": true},
            {"id": 11, "content": "<keep>", "special": true},
            {"id": 12, "content": "<left>", "lstrip": true, "rstrip": false},
            {"id": 13, "content": "<right>", "lstrip": false, "rstrip": true}
        ],
        "decoder": {"type": "ByteLevel"}
    })json";
    auto tokenizer = trtmc::CreateBpeTokenizer(source.data(), source.size(), false);
    const std::vector<std::pair<std::string, std::vector<int32_t>>> cases{
        {"a \t\n<|user|> \n\tb", {0, 10, 1}},
        {"<|user|>\n", {10}},
        {" \n<|user|>\t ", {10}},
        {"a <keep> b", {0, 2, 11, 2, 1}},
        {"a <left> b", {0, 12, 2, 1}},
        {"a <right> b", {0, 2, 13, 1}},
        {u8"a\u0085\u00a0\u1680\u2003\u2028\u2029\u202f\u205f\u3000<|user|>\u2009b", {0, 10, 1}},
        {"a \n<|user|> \t<|user|>\nb", {0, 10, 10, 1}},
        {"a \nb", {0, 2, 3, 1}},
    };
    for (const auto& [text, expected] : cases) {
        if (tokenizer->encode(text) != expected) {
            std::cerr << "FAIL: checkpoint added-token whitespace contract\n";
            return 1;
        }
    }
    return 0;
}
