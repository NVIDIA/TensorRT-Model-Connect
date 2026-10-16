/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/convbert/runtime/tokenizer.h"

#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::string json = R"json({
        "model": {"type": "WordPiece", "unk_token": "[UNK]",
                  "vocab": {"[UNK]": 0, "chases": 42, "##s": 99,
                            "extreme": 2147483647}},
        "added_tokens": [{"id": 2147483646, "content": "added", "special": false}]
    })json";
    auto tokenizer = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);
    if (tokenizer->encode("chases") != std::vector<int32_t>{42} ||
        tokenizer->id_for_token("chases") != 42 || tokenizer->token_for_id(42) != "chases" ||
        tokenizer->decode({42}) != "chases" || !tokenizer->token_for_id(1).empty() ||
        tokenizer->encode("unlisted") != std::vector<int32_t>{0} ||
        tokenizer->encode("extreme added") != std::vector<int32_t>{2147483647, 2147483646} ||
        tokenizer->decode({2147483647, 2147483646}) != "extreme added" ||
        tokenizer->id_for_token("extreme") != 2147483647 ||
        tokenizer->id_for_token("added") != 2147483646) {
        std::cerr << "Sparse vocabulary IDs must survive encode and decode\n";
        return 1;
    }
    return 0;
}
