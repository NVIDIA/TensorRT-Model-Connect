/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/jina_embeddings_v3/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

namespace {
int regressions() {
    using nlohmann::json;
    const auto fixture = json::parse(R"({
      "model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0],["<s>",0],["</s>",0],["▁",-2],["▁a",-1],["x",-1],["▁x",-1]]},
      "pre_tokenizer":{"type":"Metaspace","replacement":"▁","add_prefix_space":true,"prepend_scheme":"always"},
      "post_processor":{"type":"RobertaProcessing","cls":["<s>",1],"sep":["</s>",2]}
    })");
    auto make = [](const json& config) {
        const auto raw = config.dump();
        return trtmc::CreateUnigramTokenizer(raw.data(), raw.size(), true);
    };
    if (make(fixture)->encode("a😀🤔") != std::vector<int32_t>{1, 4, 0, 2})
        throw std::runtime_error("consecutive unknown scalars must fuse to one unknown token");
    if (make(fixture)->encode("x") != std::vector<int32_t>{1, 6, 2})
        throw std::runtime_error("always policy must prepend the metaspace marker");
    auto never = fixture;
    never["pre_tokenizer"]["prepend_scheme"] = "never";
    never["pre_tokenizer"]["add_prefix_space"] = false;
    if (make(never)->encode("x") != std::vector<int32_t>{1, 5, 2})
        throw std::runtime_error("never policy must preserve the unprefixed token");
    for (const auto& row : json::array({json::array(), json::array({"a"}), json::array({1, 0}),
                                        json::array({"a", "score"})})) {
        auto broken = fixture;
        broken["model"]["vocab"][4] = row;
        bool rejected = false;
        try {
            make(broken);
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("malformed vocabulary row must raise a load error");
    }
    auto unsupported = fixture;
    unsupported["pre_tokenizer"]["prepend_scheme"] = "first";
    bool rejected = false;
    try {
        make(unsupported);
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("unsupported metaspace policy must not be silently accepted");
    std::cout << "Jina tokenizer regressions passed\n";
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 1)
        return regressions();
    if (argc != 3)
        return 2;
    std::ifstream file(argv[1]);
    const std::string data((std::istreambuf_iterator<char>(file)), {});
    const auto tokenizer = trtmc::CreateUnigramTokenizer(data.data(), data.size(), true);
    std::cout << nlohmann::json(tokenizer->encode(argv[2])).dump() << '\n';
}
