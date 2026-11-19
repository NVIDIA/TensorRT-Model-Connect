/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Native tokenizer against Hugging Face tokenizers (tests/data/tokenizer_cases.json,
// see tests/test_fixtures.py), plus the host sampler's selection rules.
// Usage: test_smolvlm_tokenizer TOKENIZER_CASES.json
#include "families/smolvlm/runtime/sampler.h"
#include "families/smolvlm/runtime/tokenizer.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void test_tokenizer(const nlohmann::json& fixture) {
    const trtmc::smolvlm::Tokenizer tokenizer(fixture.at("tokenizer").get<std::string>());
    for (const auto& item : fixture.at("cases")) {
        const std::string text = item.at("text");
        const auto ids = tokenizer.encode(text);
        check(ids == item.at("ids").get<std::vector<int32_t>>(),
              "encode " + nlohmann::json(text).dump());
        check(tokenizer.decode(ids) == item.at("decoded").get<std::string>(),
              "decode " + nlohmann::json(text).dump());
    }
    for (const auto& item : fixture.at("decode_only"))
        check(tokenizer.decode(item.at("ids").get<std::vector<int32_t>>()) ==
                  item.at("decoded").get<std::string>(),
              "lossy decode " + item.at("ids").dump());
    check(tokenizer.added_token_id("<image>") >= 0 && tokenizer.added_token_id("<absent>") == -1,
          "added-token lookup");
    // A single pre-tokenized piece of 100,000 letters merges into 50,000 "ll" units. Merging
    // with a full rescan per merge would need about 5e9 pair lookups for it.
    const std::string letters(100000, 'l');
    const auto start = std::chrono::steady_clock::now();
    const auto pairs = tokenizer.encode(letters);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    check(pairs.size() == letters.size() / 2 && tokenizer.decode(pairs) == letters,
          "a long piece merges into pairs");
    check(seconds < 10.0, "a 100,000-byte piece took " + std::to_string(seconds) + " s");
}

void test_sampler() {
    const std::vector<float> logits{0.5F, 3.0F, -1.0F, 3.0F, 2.5F};
    const auto vocab = static_cast<int32_t>(logits.size());
    trtmc::smolvlm::Sampler greedy({1.0, 1, 1.0, -1});
    check(greedy.greedy() && greedy.next(logits.data(), vocab) == 1, "top_k 1 is first-max greedy");
    trtmc::smolvlm::Sampler cold({0.0, 0, 1.0, 7});
    check(cold.greedy() && cold.next(logits.data(), vocab) == 1, "temperature 0 is greedy");
    trtmc::smolvlm::SamplingParams sampled{1.0, 2, 1.0, 11};
    trtmc::smolvlm::Sampler first(sampled), second(sampled);
    bool same = true, within = true;
    for (int i = 0; i < 64; ++i) {
        const auto a = first.next(logits.data(), vocab), b = second.next(logits.data(), vocab);
        same = same && a == b;
        within = within && (a == 1 || a == 3);
    }
    check(same && within, "seeded top-k sampling is reproducible and stays in the top k");
    const std::vector<float> distinct{0.5F, 3.0F, -1.0F, 2.0F, 2.5F};
    trtmc::smolvlm::Sampler nucleus({1.0, 0, 1e-6, 3});
    check(nucleus.next(distinct.data(), vocab) == 1,
          "a tiny top_p keeps only the most likely token");
    for (const trtmc::smolvlm::SamplingParams bad :
         {trtmc::smolvlm::SamplingParams{-1.0, 1, 1.0, -1},
          trtmc::smolvlm::SamplingParams{1.0, -1, 1.0, -1},
          trtmc::smolvlm::SamplingParams{1.0, 1, 0.0, -1},
          trtmc::smolvlm::SamplingParams{1.0, 1, 1.5, -1},
          trtmc::smolvlm::SamplingParams{1.0, 1, 1.0, -2}}) {
        bool threw = false;
        try {
            trtmc::smolvlm::Sampler rejected(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "invalid sampling parameters are rejected");
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_smolvlm_tokenizer TOKENIZER_CASES.json\n";
        return 2;
    }
    try {
        std::ifstream input(argv[1]);
        if (!input)
            throw std::runtime_error(std::string("cannot read ") + argv[1]);
        nlohmann::json fixture;
        input >> fixture;
        test_tokenizer(fixture);
        test_sampler();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures)
        return 1;
    std::cout << "test_smolvlm_tokenizer passed\n";
    return 0;
}
