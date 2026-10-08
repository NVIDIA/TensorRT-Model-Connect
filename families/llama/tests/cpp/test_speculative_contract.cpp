/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/engine.h"

#include <iostream>
#include <stdexcept>

namespace {
template <typename F>
void expect_invalid(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("invalid contract or result was accepted");
}
} // namespace

int main() {
    using namespace trtmc::llama::speculative;
    try {
        auto manifest = nlohmann::json::parse(R"({
          "version":1,"role":"target","precision":"fp16",
          "cache_layout":"batch_heads_capacity_dim","cache_update":"aliased_contiguous_append",
          "layers":32,"hidden_size":4096,"kv_heads":8,"head_dim":128,"vocab_size":128256,
          "capacity":2048,"max_query":1024,"feature_width":12288,
          "execution_profiles":[
            {"phase":"prefill","query":[1,1024,1024],"logits":[1,1,1]},
            {"phase":"decode","query":[1,5,9],"logits":[1,5,9]}]})");
        auto contract = Contract::parse(manifest);
        if (contract.query_limit(Phase::kPrefill) != 1024 ||
            contract.query_limit(Phase::kDecode) != 9)
            throw std::runtime_error("phase query limits are not independent");
        manifest["execution_profiles"][0]["phase"] = "decode";
        expect_invalid([&] { Contract::parse(manifest); });
        manifest["execution_profiles"][0]["phase"] = "prefill";
        manifest["execution_profiles"][1]["query"] = {1, 5, 9, 64};
        expect_invalid([&] { Contract::parse(manifest); });
        manifest.erase("execution_profiles");
        manifest["max_query"] = 64;
        contract = Contract::parse(manifest);
        if (contract.query_limit(Phase::kPrefill) != 64 ||
            contract.query_limit(Phase::kDecode) != 64)
            throw std::runtime_error("single-profile contract changed");
        manifest["version"] = 2;
        expect_invalid([&] { Contract::parse(manifest); });
        manifest["version"] = 1;
        manifest["role"] = "draft";
        manifest["vocab_size"] = 1;
        expect_invalid([&] { Contract::parse(manifest); });

        // GPU selection/path algorithms have independent oracle tests in
        // selection_gpu.py and device_policy_gpu.py. Check the readback ABI here.
        StepResult result;
        result.vocab = 4;
        result.ranks = 1;
        result.selection = {2, 1, 0, 0, 1, 1};
        if (result.token() != 2 || result.token(2) != 1)
            throw std::runtime_error("compact selection row order changed");
        expect_invalid([&] { result.token(1); }); // consumed non-finite row
        expect_invalid([&] { result.token(-1); });
        expect_invalid([&] { result.token(3); });
        expect_invalid([&] { result.token(0, 1); });
        result.ranks = 2;
        result.selection = {1, 2, 1};
        if (result.token() != 1 || result.token(0, 1) != 2)
            throw std::runtime_error("compact draft rank order changed");
        result.selection[0] = 4;
        expect_invalid([&] { result.token(); });
        result.selection[0] = -1;
        expect_invalid([&] { result.token(); });
        result.selection.clear();
        expect_invalid([&] { result.token(); });

        std::uint16_t feature_data[12]{};
        FeatureView features{feature_data, 3, 4};
        const auto last = features.slice(2, 1);
        if (last.data != feature_data + 8 || last.rows != 1 || last.width != 4)
            throw std::runtime_error("device feature row offset changed");
        expect_invalid([&] { features.slice(2, 2); });
        std::cout << "speculative contract tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
