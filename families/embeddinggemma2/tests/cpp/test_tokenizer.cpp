/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/embeddinggemma2/runtime/tokenizer.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        const std::string json = R"({
          "model": {"type":"BPE", "vocab":{"<unk>":0,"<bos>":2,"<eos>":1,"\u2581":3,"a":4}, "merges":[]},
          "added_tokens":[{"id":2,"content":"<bos>","special":true}, {"id":1,"content":"<eos>","special":true}],
          "normalizer":{"type":"Replace", "pattern":{"String":" "}, "content":"\u2581"},
          "decoder":{"type":"Sequence", "decoders":[{"type":"Replace", "pattern":{"String":"\u2581"}, "content":" "}]},
          "pre_tokenizer":{"type":"Split", "pattern":{"String":" "}, "behavior":"MergedWithPrevious", "invert":false},
          "post_processor":{"type":"TemplateProcessing", "single":[{"SpecialToken":{"id":"<bos>"}}, {"Sequence":{"id":"A"}}, {"SpecialToken":{"id":"<eos>"}}]}
        })";
        const auto tokenizer = trtmc::CreateBpeTokenizer(json.data(), json.size(), true);
        if (tokenizer->encode("") != std::vector<int32_t>{2, 1})
            throw std::runtime_error("empty text lost BOS/EOS framing");
        const auto ids = tokenizer->encode(" a");
        if (ids != std::vector<int32_t>{2, 3, 4, 1})
            throw std::runtime_error("space replacement or BOS/EOS framing mismatch");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
