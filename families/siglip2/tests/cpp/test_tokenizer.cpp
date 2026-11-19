/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The SigLIP 2 tokenizer contract on a miniature tokenizer.json of the same shape: a
// byte-fallback BPE whose only normaliser is the space marker, no pre-tokenizer, and a
// template that appends <eos>. The real tokenizer.json is exercised by the E2E, whose text
// embeddings are compared with the reference model's.

#include "families/siglip2/runtime/tokenizer.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    try {
        const std::string json = R"({
          "added_tokens":[{"id":0,"content":"<pad>","special":true},
                          {"id":1,"content":"<eos>","special":true},
                          {"id":2,"content":"<bos>","special":true}],
          "normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
          "pre_tokenizer":null,
          "post_processor":{"type":"TemplateProcessing",
                            "single":[{"Sequence":{"id":"A","type_id":0}},
                                      {"SpecialToken":{"id":"<eos>","type_id":0}}],
                            "special_tokens":{"<eos>":{"id":"<eos>","ids":[1],"tokens":["<eos>"]}}},
          "decoder":{"type":"Sequence","decoders":[
                       {"type":"Replace","pattern":{"String":"▁"},"content":" "},
                       {"type":"ByteFallback"},{"type":"Fuse"}]},
          "model":{"type":"BPE","byte_fallback":true,"fuse_unk":true,"unk_token":"<unk>",
                   "vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,"▁":4,"a":5,"b":6,
                            "ab":7,"▁ab":8,"<0xC3>":9,"<0xA9>":10,"A":11,
                            "<0xE2>":12,"<0x82>":13,"<0xF0>":14,"<0x9F>":15},
                   "merges":[["a","b"],["▁","ab"]]}
        })";
        const auto tokenizer = trtmc::CreateSiglip2Tokenizer(json.data(), json.size(), true);
        if (tokenizer->encode("") != std::vector<int32_t>{1})
            throw std::runtime_error("empty text must still carry the end token");
        // No leading marker is prepended: "ab" is one piece, " ab" merges with the marker.
        if (tokenizer->encode("ab") != std::vector<int32_t>{7, 1})
            throw std::runtime_error("plain piece or end token mismatch");
        if (tokenizer->encode("a ab") != std::vector<int32_t>{5, 8, 1})
            throw std::runtime_error("space marker merge mismatch");
        // Case is preserved: the reference tokenizer for this checkpoint does not lowercase.
        if (tokenizer->encode("A") != std::vector<int32_t>{11, 1})
            throw std::runtime_error("upper case was changed");
        // A character outside the vocabulary falls back to its UTF-8 bytes.
        if (tokenizer->encode("\xc3\xa9") != std::vector<int32_t>{9, 10, 1})
            throw std::runtime_error("byte fallback mismatch");
        // An added token in the text is matched as itself.
        if (tokenizer->encode("a<eos>") != std::vector<int32_t>{5, 1, 1})
            throw std::runtime_error("added token mismatch");
        // A prompt that ends inside a multi-byte sequence keeps the per-byte fallback, and the
        // reader does not step past the end of the text to find that out.
        std::string failures;
        const auto expect = [&](const std::string& text, const std::vector<int32_t>& ids,
                                const char* what) {
            const auto got = tokenizer->encode(text);
            if (got != ids) {
                failures += std::string(what) + ": got";
                for (const auto id : got)
                    failures += " " + std::to_string(id);
                failures += "\n";
            }
        };
        expect("a\xc3", {5, 9, 1}, "truncated two-byte sequence");
        expect("a\xe2\x82", {5, 12, 13, 1}, "truncated three-byte sequence");
        expect("a\xe2", {5, 12, 1}, "three-byte lead at the end");
        expect("a\xf0\x9f", {5, 14, 15, 1}, "truncated four-byte sequence");
        expect("a\xf0", {5, 14, 1}, "four-byte lead at the end");
        if (!failures.empty())
            throw std::runtime_error(failures);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
