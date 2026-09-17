/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bart/runtime/tokenizer.h"

#include <iostream>
#include <string>
#include <vector>

int main() {
    // BART's <mask> consumes preceding whitespace; its other special tokens do not.
    const std::string tokenizer_json = R"json({
        "model": {"type": "BPE", "vocab": {
            "<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
            "H": 4, "i": 5, "\u0120": 6, "<mask>": 7,
            "\u0109": 8, "\u010a": 9, "\u010b": 10, "\u010c": 11, "\u010d": 12,
            "\u0122": 13, "\u0123": 14, "\u0124": 15, "\u0125": 16, "\u0126": 17,
            "\u0127": 18, "\u0128": 19, "\u0129": 20, "\u012a": 21, "\u012b": 22,
            "\u012c": 23, "\u012d": 24, "\u013c": 25, "\u0141": 26, "\u0142": 27,
            "\u00a8": 28, "\u00a9": 29, "\u00af": 30, "\u00c2": 31,
            "\u00e1": 32, "\u00e2": 33, "\u00e3": 34
        }, "merges": []},
        "added_tokens": [
            {"id": 0, "content": "<s>", "special": true, "lstrip": false},
            {"id": 2, "content": "</s>", "special": true},
            {"id": 7, "content": "<mask>", "special": true, "lstrip": true}
        ],
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "post_processor": {"type": "RobertaProcessing", "cls": ["<s>", 0], "sep": ["</s>", 2]},
        "decoder": {"type": "ByteLevel"}
    })json";
    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), true);
    int failures = 0;
    const auto check = [&](const std::string& text, const std::vector<int32_t>& expected) {
        if (tokenizer->encode(text) != expected) {
            std::cerr << "Unexpected token IDs for: " << text << '\n';
            ++failures;
        }
    };

    check("Hi <mask> Hi", {0, 4, 5, 7, 6, 4, 5, 2});
    check("Hi<mask> Hi", {0, 4, 5, 7, 6, 4, 5, 2});
    check("Hi   <mask>", {0, 4, 5, 7, 2});
    check("Hi \t\r\n<mask>", {0, 4, 5, 7, 2});
    check("   <mask>", {0, 7, 2});
    check("<mask> <mask>", {0, 7, 7, 2});
    check("Hi <s> Hi", {0, 4, 5, 6, 0, 6, 4, 5, 2});
    check("Hi </s>", {0, 4, 5, 6, 2, 2});
    check("Hi Hi", {0, 4, 5, 6, 4, 5, 2});
    check(u8"Hi\u200b<mask>", {0, 4, 5, 33, 13, 24, 7, 2});

    const std::vector<std::string> whitespace = {
        u8"\u0085", u8"\u00a0", u8"\u1680", u8"\u2000", u8"\u2001", u8"\u2002", u8"\u2003",
        u8"\u2004", u8"\u2005", u8"\u2006", u8"\u2007", u8"\u2008", u8"\u2009", u8"\u200a",
        u8"\u2028", u8"\u2029", u8"\u202f", u8"\u205f", u8"\u3000",
    };
    for (const auto& space : whitespace)
        check("Hi" + space + "<mask>", {0, 4, 5, 7, 2});

    auto without_special_tokens =
        trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    if (without_special_tokens->encode("Hi <mask>") != std::vector<int32_t>{4, 5, 7}) {
        std::cerr << "lstrip must also apply when post-processing is disabled\n";
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}
