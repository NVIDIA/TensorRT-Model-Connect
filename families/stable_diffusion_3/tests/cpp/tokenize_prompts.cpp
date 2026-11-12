/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Frames prompts with the family's runtime tokenizers, exactly as the
// pipeline does, and prints the ids as JSON for tests/test_e2e.py.
//
// Usage: test_stable_diffusion_3_tokenize <request.json>
//   request: {"clip_l": {"tokenizer": path, "bos": n, "eos": n, "pad": n, "config_eos": n},
//             "clip_g": {...}, "t5": {"tokenizer": path, "eos": n, "pad": n},
//             "clip_length": 77, "t5_length": 256, "prompts": ["..."]}
//   output:  {"clip_l": [{"ids": [...], "pool": n}, ...], "clip_g": [...], "t5": [[...], ...]}

#include "families/stable_diffusion_3/runtime/sampling.h"
#include "families/stable_diffusion_3/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

namespace sd3 = trtmc::stable_diffusion_3;

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read " + path);
    std::stringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

nlohmann::json clip_ids(const nlohmann::json& spec, const nlohmann::json& prompts,
                        std::size_t length) {
    const auto data = read_file(spec.at("tokenizer").get<std::string>());
    const auto tokenizer = trtmc::CreateBpeTokenizer(data.data(), data.size(), false);
    nlohmann::json out = nlohmann::json::array();
    for (const auto& prompt : prompts) {
        const auto ids =
            sd3::frame_clip_ids(tokenizer->encode(prompt.get<std::string>()), spec.at("bos"),
                                spec.at("eos"), spec.at("pad"), length);
        out.push_back({{"ids", ids}, {"pool", sd3::clip_pool_index(ids, spec.at("config_eos"))}});
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <request.json>\n";
        return 2;
    }
    try {
        const auto request = nlohmann::json::parse(read_file(argv[1]));
        const auto& prompts = request.at("prompts");
        const auto clip_length = request.at("clip_length").get<std::size_t>();
        nlohmann::json out;
        out["clip_l"] = clip_ids(request.at("clip_l"), prompts, clip_length);
        out["clip_g"] = clip_ids(request.at("clip_g"), prompts, clip_length);
        const auto& t5 = request.at("t5");
        const auto data = read_file(t5.at("tokenizer").get<std::string>());
        const auto tokenizer = trtmc::CreateUnigramTokenizer(data.data(), data.size(), false);
        out["t5"] = nlohmann::json::array();
        for (const auto& prompt : prompts)
            out["t5"].push_back(sd3::frame_t5_ids(tokenizer->encode(prompt.get<std::string>()),
                                                  t5.at("eos"), t5.at("pad"),
                                                  request.at("t5_length").get<std::size_t>()));
        std::cout << out.dump() << "\n";
    } catch (const std::exception& error) {
        std::cerr << "tokenize_prompts: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
