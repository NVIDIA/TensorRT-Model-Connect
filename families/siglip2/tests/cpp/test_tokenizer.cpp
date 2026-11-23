/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The SigLIP 2 tokenizer contract on a miniature tokenizer.json of the same shape: a
// byte-fallback BPE whose only normaliser is the space marker, no pre-tokenizer, and a
// template that appends <eos>. The real tokenizer.json is exercised by the E2E, whose text
// embeddings are compared with the reference model's.

#include "families/siglip2/runtime/tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

constexpr std::int32_t kEos = 1;

// Like the checkpoint's vocabulary: the pieces, then one byte token per byte in byte
// order, except that the tab's slot holds the tab itself.
std::int32_t byte_id(int byte) {
    return 11 + byte;
}

Json added_token(std::int32_t id, const std::string& content, bool special,
                 const char* option = nullptr) {
    Json token = {{"id", id},          {"content", content}, {"single_word", false},
                  {"lstrip", false},   {"rstrip", false},    {"normalized", false},
                  {"special", special}};
    if (option != nullptr)
        token[option] = true;
    return token;
}

Json miniature() {
    Json tokenizer = Json::parse(R"({
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
               "vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,"▁":4,"a":5,"b":6,"ab":7,
                        "▁ab":8,"A":9,"aba":10},
               "merges":[["a","b"],["▁","ab"]]}
    })");
    tokenizer["added_tokens"] =
        Json::array({added_token(0, "<pad>", true), added_token(1, "<eos>", true),
                     added_token(2, "<bos>", true)});
    auto& vocab = tokenizer["model"]["vocab"];
    for (int byte = 0; byte < 256; ++byte) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", byte);
        vocab[byte == 0x09 ? std::string("\t") : std::string(name)] = byte_id(byte);
    }
    return tokenizer;
}

// Renames a vocabulary entry, keeping its id, so that the ids stay 0..N-1.
void rename_token(Json& tokenizer, const std::string& from, const std::string& to) {
    auto& vocab = tokenizer["model"]["vocab"];
    vocab[to] = vocab[from];
    vocab.erase(from);
}

std::unique_ptr<trtmc::ITokenizer> load(const Json& tokenizer) {
    const std::string text = tokenizer.dump();
    return trtmc::CreateSiglip2Tokenizer(text.data(), text.size(), true);
}

struct Refusal {
    const char* what;
    std::function<void(Json&)> change;
    const char* message;
};

} // namespace

int main() {
    std::string failures;
    try {
        const auto tokenizer = load(miniature());
        const auto expect = [&](const std::string& text, const std::vector<std::int32_t>& ids,
                                const char* what) {
            const auto got = tokenizer->encode(text);
            if (got != ids) {
                failures += std::string(what) + ": got";
                for (const auto id : got)
                    failures += " " + std::to_string(id);
                failures += "\n";
            }
        };
        expect("", {kEos}, "empty text must still carry the end token");
        // No leading marker is prepended: "ab" is one piece, " ab" merges with the marker.
        expect("ab", {7, kEos}, "plain piece");
        expect("a ab", {5, 8, kEos}, "space marker merge");
        // Case is preserved: the reference tokenizer for this checkpoint does not lowercase.
        expect("A", {9, kEos}, "upper case");
        // A character outside the vocabulary falls back to its UTF-8 bytes; the tab, whose
        // byte token the vocabulary leaves out, is an entry of its own.
        expect("\xc3\xa9", {byte_id(0xC3), byte_id(0xA9), kEos}, "byte fallback");
        expect("\t", {byte_id(0x09), kEos}, "tab");
        // An added token in the text is matched as itself.
        expect("a<eos>", {5, kEos, kEos}, "added token");
        // A prompt that ends inside a multi-byte sequence keeps the per-byte fallback, and the
        // reader does not step past the end of the text to find that out.
        expect("a\xc3", {5, byte_id(0xC3), kEos}, "truncated two-byte sequence");
        expect("a\xe2\x82", {5, byte_id(0xE2), byte_id(0x82), kEos},
               "truncated three-byte sequence");
        expect("a\xe2", {5, byte_id(0xE2), kEos}, "three-byte lead at the end");
        expect("a\xf0\x9f", {5, byte_id(0xF0), byte_id(0x9F), kEos},
               "truncated four-byte sequence");
        expect("a\xf0", {5, byte_id(0xF0), kEos}, "four-byte lead at the end");
    } catch (const std::exception& error) {
        failures += std::string("miniature tokenizer: ") + error.what() + "\n";
    }

    // Merges run one at a time, so a pair that a merge forms is merged before the waiting
    // pairs it outranks. With (ab, a) ranked before (a, b), the reference tokenises "abab"
    // as "aba" + "b", where merging every (a, b) at once would give "ab" + "ab".
    try {
        Json ranked = miniature();
        ranked["model"]["merges"] = Json::array(
            {Json::array({"ab", "a"}), Json::array({"a", "b"}), Json::array({"▁", "ab"})});
        const auto tokenizer = load(ranked);
        if (tokenizer->encode("abab") != std::vector<std::int32_t>{10, 6, kEos})
            failures += "a pair formed by a merge did not take its rank\n";
        if (tokenizer->encode("a ab") != std::vector<std::int32_t>{5, 8, kEos})
            failures += "merge order: space marker merge\n";
    } catch (const std::exception& error) {
        failures += std::string("merge order: ") + error.what() + "\n";
    }

    // What the runtime would tokenise differently from the reference is refused on load (the
    // builder refuses it first), and so is a malformed added-token record.
    const std::vector<Refusal> refusals = {
        // The reference turns a character with a missing byte token into the unknown token.
        {"byte token missing", [](Json& t) { rename_token(t, "<0xAC>", "zz"); }, "<0xAC>"},
        {"ASCII byte token missing", [](Json& t) { rename_token(t, "<0x7A>", "zz"); }, "<0x7A>"},
        {"no byte fallback", [](Json& t) { t["model"]["byte_fallback"] = false; },
         "fall back to bytes"},
        {"single_word",
         [](Json& t) { t["added_tokens"].push_back(added_token(7, "ab", false, "single_word")); },
         "sets single_word"},
        {"lstrip",
         [](Json& t) { t["added_tokens"].push_back(added_token(7, "ab", false, "lstrip")); },
         "sets lstrip"},
        {"rstrip",
         [](Json& t) { t["added_tokens"].push_back(added_token(7, "ab", false, "rstrip")); },
         "sets rstrip"},
        {"normalized",
         [](Json& t) { t["added_tokens"].push_back(added_token(7, "ab", false, "normalized")); },
         "sets normalized"},
        // The reference gives an added token in the vocabulary the vocabulary's id.
        {"record id is not the vocabulary's",
         [](Json& t) { t["added_tokens"].push_back(added_token(5, "ab", false)); },
         "not the vocabulary entry with id 5"},
        {"added token outside the vocabulary",
         [](Json& t) { t["added_tokens"].push_back(added_token(300, "<extra>", true)); },
         "not the vocabulary entry with id 300"},
        {"template token not special", [](Json& t) { t["added_tokens"][1]["special"] = false; },
         "not a special added token"},
        // The reference appends every id the template token lists.
        {"template token expands to two ids",
         [](Json& t) {
             t["post_processor"]["special_tokens"]["<eos>"]["ids"] = Json::array({2, 1});
         },
         "exactly its id 1"},
        {"template token expands to another id",
         [](Json& t) { t["post_processor"]["special_tokens"]["<eos>"]["ids"] = Json::array({2}); },
         "exactly its id 1"},
        {"template without special_tokens",
         [](Json& t) { t["post_processor"].erase("special_tokens"); }, "exactly its id 1"},
        {"record without a flag", [](Json& t) { t["added_tokens"][2].erase("normalized"); },
         "no boolean normalized"},
        {"record without content",
         [](Json& t) {
             auto record = added_token(7, "ab", false);
             record.erase("content");
             t["added_tokens"].push_back(record);
         },
         "no string content"},
        {"record without id",
         [](Json& t) {
             auto record = added_token(7, "ab", false);
             record.erase("id");
             t["added_tokens"].push_back(record);
         },
         "ab has no valid id"},
        {"record with a negative id",
         [](Json& t) { t["added_tokens"].push_back(added_token(-1, "ab", false)); },
         "ab has no valid id"},
    };
    for (const auto& refusal : refusals) {
        Json tokenizer = miniature();
        refusal.change(tokenizer);
        try {
            load(tokenizer);
            failures += std::string(refusal.what) + ": loaded\n";
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()).find(refusal.message) == std::string::npos)
                failures += std::string(refusal.what) + ": " + error.what() + "\n";
        } catch (const std::exception& error) {
            failures += std::string(refusal.what) + ": " + error.what() + "\n";
        }
    }

    if (!failures.empty()) {
        std::cerr << failures;
        return 1;
    }
    return 0;
}
