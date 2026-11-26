/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Diff the runtime tokenizer against the reference's, on a built bundle's own asset.
//
//   tipsv2_tokenizer_check BUNDLE CASES.json
//
// `CASES.json` is `{"cases": [{"text": "...", "ids": [...]}, ...]}`, written by
// `families/tipsv2/tests/test_e2e.py` from the reference SentencePiece processor. Exits non-zero
// on the first disagreement and prints one JSON line either way, so the Python side can report
// the counts it saw.

#include "families/tipsv2/runtime/tokenizer.h"
#include "trtmc/bundle.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(std::string("cannot open ") + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

std::string join(const std::vector<std::int32_t>& ids) {
    std::string out;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        out += (index == 0 ? "" : ", ") + std::to_string(ids[index]);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: tipsv2_tokenizer_check BUNDLE CASES.json\n";
        return 2;
    }
    try {
        const trtmc::BundleReader bundle(argv[1]);
        const auto asset = bundle.read_section("tokenizer.spm");
        trtmc::tipsv2::Tokenizer tokenizer(
            trtmc::tipsv2::parse_tokenizer_asset(std::string_view(asset.data(), asset.size())));
        const auto document = nlohmann::json::parse(read_file(argv[2]));
        const auto& cases = document.at("cases");
        if (!cases.is_array() || cases.empty()) {
            std::cerr << "tokenizer_check: cases.json has no cases\n";
            return 2;
        }
        std::size_t checked = 0;
        for (const auto& entry : cases) {
            const auto text = entry.at("text").get<std::string>();
            const auto expected = entry.at("ids").get<std::vector<std::int32_t>>();
            const auto actual = tokenizer.encode(text);
            if (actual != expected) {
                nlohmann::json failure{{"status", "mismatch"},
                                       {"text", text},
                                       {"reference_ids", expected},
                                       {"native_ids", actual},
                                       {"checked", checked}};
                std::cout << failure.dump() << "\n";
                std::cerr << "tokenizer_check: " << text << "\n  reference [" << join(expected)
                          << "]\n  native    [" << join(actual) << "]\n";
                return 1;
            }
            ++checked;
        }
        nlohmann::json ok{{"status", "ok"},
                          {"checked", checked},
                          {"pieces", tokenizer.asset().pieces.size()},
                          {"longest_piece_bytes", tokenizer.longest_piece_bytes()}};
        std::cout << ok.dump() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tokenizer_check: " << error.what() << "\n";
        return 2;
    }
}
