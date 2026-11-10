/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/siglip/runtime/tokenizer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

using Json = nlohmann::json;

Json read_json(const char* path) {
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error(std::string("Cannot read fixture: ") + path);
    return Json::parse(stream);
}

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3)
            throw std::runtime_error("Expected fixture path and optional tokenizer.json path");
        const auto fixture = read_json(argv[1]);
        const auto tokenizer_json = argc == 3 ? read_json(argv[2]) : fixture.at("tokenizer");
        const auto data = tokenizer_json.dump();
        for (bool add_special : {false, true}) {
            const auto tokenizer = trtmc::CreateSiglipTokenizer(data.data(), data.size(), add_special);
            for (const auto& test : fixture.at("cases")) {
                const auto text = test.at("text").get<std::string>();
                const auto expected = test.at(add_special ? "ids_special" : "ids").get<std::vector<std::int32_t>>();
                const auto actual = tokenizer->encode(text);
                if (actual != expected)
                    throw std::runtime_error("Encode mismatch: " + test.dump() + " actual=" + Json(actual).dump());
                if (tokenizer->decode(actual) != test.at("decoded").get<std::string>())
                    throw std::runtime_error("Decode mismatch: " + test.dump());
            }
            for (const auto& test : fixture.at("decode_cases")) {
                if (tokenizer->decode(test.at("ids").get<std::vector<std::int32_t>>()) !=
                    test.at("decoded").get<std::string>())
                    throw std::runtime_error("Byte fallback decode mismatch: " + test.dump());
            }
            if (tokenizer->id_for_token("<pad>") != 0 || tokenizer->id_for_token("absent-token") != -1 ||
                tokenizer->token_for_id(-1) != "" || tokenizer->token_for_id(0) != "<pad>")
                throw std::runtime_error("Token lookup mismatch");
        }
        for (const char* field : {"normalizer", "pre_tokenizer", "decoder", "post_processor"}) {
            auto invalid = tokenizer_json;
            invalid[field] = nullptr;
            const auto invalid_data = invalid.dump();
            bool refused = false;
            try {
                trtmc::CreateSiglipTokenizer(invalid_data.data(), invalid_data.size(), true);
            } catch (const std::exception&) {
                refused = true;
            }
            if (!refused)
                throw std::runtime_error(std::string("Accepted unsupported ") + field);
        }
        if (argc == 2) {
            const auto unigram = Json::parse(R"({
                "model":{"type":"Unigram","unk_id":2,"vocab":[
                    ["<pad>",0],["</s>",0],["<unk>",-100],["▁ab",-1],["▁c",-1]]},
                "normalizer":{"type":"Lowercase"}
            })").dump();
            const auto tokenizer = trtmc::CreateSiglipTokenizer(unigram.data(), unigram.size(), true);
            if (tokenizer->encode("AB C") != std::vector<std::int32_t>{3, 4, 1} ||
                tokenizer->decode({3, 4, 1}) != "ab c" ||
                tokenizer->encode("") != std::vector<std::int32_t>{1})
                throw std::runtime_error("Unigram dispatch regression");
        }
        std::cout << "siglip BPE: " << fixture.at("cases").size()
                  << " encode cases with/without EOS, decode and refusal checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
