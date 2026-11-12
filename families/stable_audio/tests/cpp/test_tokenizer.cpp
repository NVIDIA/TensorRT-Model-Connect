/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The family's T5 unigram tokenizer against the tokenizers library (0.22.2): a small
// T5-style tokenizer.json (no normalizer, WhitespaceSplit + Metaspace, Unigram with
// <unk> id 2) and the ids tokenizers returns for each prompt without special tokens.
// Covers whole-word and split pieces, repeated whitespace, and runs of unknown
// characters (full-width letters, CJK, emoji), which the Unigram model fuses into one
// <unk>.

#include "families/stable_audio/runtime/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

const char kTokenizerJson[] =
    R"json({"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<pad>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":1,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always","split":true}]},"post_processor":{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"</s>","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"</s>","type_id":0}},{"Sequence":{"id":"B","type_id":0}},{"SpecialToken":{"id":"</s>","type_id":0}}],"special_tokens":{"</s>":{"id":"</s>","ids":[1],"tokens":["</s>"]}}},"decoder":{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always","split":true},"model":{"type":"Unigram","unk_id":2,"vocab":[["<pad>",0.0],["</s>",0.0],["<unk>",0.0],["\u2581",-2.5],["\u2581a",-1.5],["a",-3.0],["b",-3.0],["\u2581ab",-1.2],["c",-4.0],["\u2581dr",-2.0],["um",-2.2],["s",-2.4],["\u2581drums",-1.1],["\u2581the",-1.0],["\u2581rain",-1.3],["!",-3.5],[".",-3.1]],"byte_fallback":false}})json";

} // namespace

int main() {
    const std::string json(kTokenizerJson);
    auto tokenizer = trtmc::stable_audio::CreateUnigramTokenizer(json.data(), json.size(), false);
    if (!tokenizer) {
        std::fprintf(stderr, "tokenizer.json did not load\n");
        return EXIT_FAILURE;
    }
    const std::vector<std::pair<std::string, std::vector<std::int32_t>>> cases = {
        {"the drums", {13, 12}},
        {"a b ab", {4, 3, 6, 7}},
        {"the rain!", {13, 14, 15}},
        {"Drums", {3, 2, 10, 11}},
        {"abcab.", {7, 8, 5, 6, 16}},
        {"ab  \x09"
         " the   rain",
         {7, 13, 14}},
        {"\xef"
         "\xbc"
         "\xa6"
         "\xef"
         "\xbd"
         "\x95"
         "\xef"
         "\xbd"
         "\x8c"
         "\xef"
         "\xbd"
         "\x8c"
         " drums",
         {3, 2, 12}},
        {"\xe6"
         "\x97"
         "\xa5"
         "\xe6"
         "\x9c"
         "\xac"
         "\xe3"
         "\x81"
         "\xae"
         "\xe5"
         "\xa4"
         "\xaa"
         "\xe9"
         "\xbc"
         "\x93"
         "",
         {3, 2}},
        {"\xf0"
         "\x9f"
         "\x8e"
         "\xb5"
         "\xf0"
         "\x9f"
         "\x8e"
         "\xb5"
         " the drums",
         {3, 2, 13, 12}},
        {"x y z", {3, 2, 3, 2, 3, 2}},
        {"", {}},
    };
    int failures = 0;
    for (const auto& [prompt, expected] : cases) {
        const auto ids = tokenizer->encode(prompt);
        if (ids != expected) {
            std::fprintf(stderr, "prompt \"%s\": %zu ids, expected %zu\n", prompt.c_str(),
                         ids.size(), expected.size());
            ++failures;
        }
    }
    if (failures != 0) {
        std::fprintf(stderr, "stable_audio tokenizer: %d mismatches\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("stable_audio tokenizer: %zu prompts match tokenizers\n", cases.size());
    return EXIT_SUCCESS;
}
