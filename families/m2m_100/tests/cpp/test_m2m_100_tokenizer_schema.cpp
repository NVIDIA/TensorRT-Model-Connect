/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/m2m_100/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string tokenizer_json =
        R"json({"normalizer":{"type":"Sequence","normalizers":[{"type":"Precompiled"}]},"pre_tokenizer":{"type":"Metaspace","replacement":"▁","add_prefix_space":true},"post_processor":{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"</s>","type_id":0}}]},"added_tokens":[{"id":11,"content":"</s>","special":true}],"model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":true,"vocab":{"<unk>":0,"▁":1,"h":2,"e":3,"l":4,"o":5,"▁h":6,"▁he":7,"▁hel":8,"▁hell":9,"▁hello":10,"</s>":11},"merges":["▁ h","▁h e","▁he l","▁hel l","▁hell o"]}})json";
    std::string input = "hello";
    std::vector<std::int32_t> expected{10, 11};
    if (argc == 2) {
        std::ifstream stream(argv[1]);
        tokenizer_json.assign(std::istreambuf_iterator<char>(stream),
                              std::istreambuf_iterator<char>());
        input = "The house is wonderful.";
        expected = {1617, 34151, 248, 157826, 248075, 2, 3};
    }
    auto tokenizer =
        trtmc::CreateSentencePieceBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), true);
    if (!tokenizer) {
        std::cerr << "native SentencePiece-BPE tokenizer rejected its object vocab schema\n";
        return 1;
    }
    if (tokenizer->encode(input) != expected) {
        std::cerr << "native SentencePiece-BPE tokenizer did not apply the declared merges\n";
        return 1;
    }
    return 0;
}
