/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/fnet/runtime/tokenizer.h"

#include <iostream>

int main() {
    const std::string source = R"json({
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "Ġ": 2, "ĉ": 3}, "merges": []},
        "normalizer": {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
        "decoder": {"type": "ByteLevel"}
    })json";
    auto tokenizer = trtmc::CreateBpeTokenizer(source.data(), source.size(), false);
    if (tokenizer->encode("a  b") != tokenizer->encode("a b") ||
        tokenizer->encode("a \tb") != std::vector<int32_t>{0, 2, 3, 1}) {
        std::cerr << "FAIL: normalize repeated spaces without collapsing other whitespace\n";
        return 1;
    }
    return 0;
}
