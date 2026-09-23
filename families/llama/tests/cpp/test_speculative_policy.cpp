/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/engine.h"

#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using trtmc::llama::speculative::argmax;
    using trtmc::llama::speculative::greedy_path;
    try {
        const float tied_logits[]{-3.0F, 7.0F, 7.0F, -1.0F};
        const float negative_logits[]{-5.0F, -1.0F, -3.0F};
        const float zero_logits[]{0.0F, -0.0F, 0.0F};
        if (argmax(tied_logits, 4) != 1 || argmax(negative_logits, 3) != 1 ||
            argmax(zero_logits, 3) != 0 || argmax(negative_logits, 1) != 0)
            throw std::runtime_error("argmax changed finite values or first-index tie breaking");
        for (const float invalid :
             {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity()}) {
            for (const int position : {0, 1, 3}) {
                std::vector<float> values(tied_logits, tied_logits + 4);
                values[position] = invalid;
                bool rejected = false;
                try {
                    argmax(values.data(), static_cast<int>(values.size()));
                } catch (const std::invalid_argument&) {
                    rejected = true;
                }
                if (!rejected)
                    throw std::runtime_error("argmax accepted non-finite logits");
            }
        }
        for (const int count : {0, -1}) {
            bool rejected = false;
            try {
                argmax(nullptr, count);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("argmax accepted an empty logit range");
        }
        using trtmc::llama::speculative::Contract;
        using trtmc::llama::speculative::Phase;
        auto manifest = nlohmann::json::parse(R"({
          "version":1,"role":"target","precision":"fp16",
          "cache_layout":"batch_heads_capacity_dim","cache_update":"aliased_contiguous_append",
          "layers":32,"hidden_size":4096,"kv_heads":8,"head_dim":128,"vocab_size":128256,
          "capacity":2048,"max_query":1024,"feature_width":12288,
          "execution_profiles":[
            {"phase":"prefill","query":[1,1024,1024],"logits":[1,1,1]},
            {"phase":"decode","query":[1,5,9],"logits":[1,5,9]}]})");
        auto contract = Contract::parse(manifest);
        if (contract.device_selection)
            throw std::runtime_error("legacy bundle enabled device selection implicitly");
        manifest["greedy_selection"] = "device_v1";
        if (!Contract::parse(manifest).device_selection)
            throw std::runtime_error("device selection capability was ignored");
        manifest["greedy_selection"] = "unknown";
        bool unknown_selection = false;
        try {
            Contract::parse(manifest);
        } catch (const std::invalid_argument&) {
            unknown_selection = true;
        }
        if (!unknown_selection)
            throw std::runtime_error("unknown selection capability was accepted");
        manifest.erase("greedy_selection");
        if (contract.query_limit(Phase::kPrefill) != 1024 ||
            contract.query_limit(Phase::kDecode) != 9)
            throw std::runtime_error("phase query limits are not independent");
        manifest["execution_profiles"][0]["phase"] = "decode";
        bool bad_profile = false;
        try {
            Contract::parse(manifest);
        } catch (const std::invalid_argument&) {
            bad_profile = true;
        }
        if (!bad_profile)
            throw std::runtime_error("accepted swapped execution profiles");
        manifest["execution_profiles"][0]["phase"] = "prefill";
        manifest["execution_profiles"][1]["query"] = {1, 5, 9, 64};
        bad_profile = false;
        try {
            Contract::parse(manifest);
        } catch (const std::invalid_argument&) {
            bad_profile = true;
        }
        if (!bad_profile)
            throw std::runtime_error("accepted malformed profile bounds");
        manifest.erase("execution_profiles");
        manifest["max_query"] = 64;
        contract = Contract::parse(manifest);
        if (contract.query_limit(Phase::kPrefill) != 64 ||
            contract.query_limit(Phase::kDecode) != 64)
            throw std::runtime_error("legacy single-profile contract changed");
        manifest["version"] = 2;
        bool unsupported_version = false;
        try {
            Contract::parse(manifest);
        } catch (const std::invalid_argument&) {
            unsupported_version = true;
        }
        if (!unsupported_version)
            throw std::runtime_error("accepted an unsupported state ABI version");
        // Greedy target picks sibling row 2, then its child row 4. Physical
        // row order differs from the logical accepted path.
        const std::vector<std::int32_t> tokens{0, 1, 2, 3, 1};
        const std::vector<std::int32_t> parents{-1, 0, 0, 1, 2};
        std::vector<float> logits(5 * 4, 0);
        logits[2] = 4;
        logits[2 * 4 + 1] = 4;
        if (greedy_path(tokens, parents, logits, 4) != std::vector<std::int32_t>({0, 2, 4}))
            throw std::runtime_error("branch acceptance chose the wrong path");
        trtmc::llama::speculative::StepResult compact;
        compact.vocab = 4;
        compact.ranks = 1;
        compact.selection = {2, 1, 0, 0, 1, 1, 0, 0, 0, 1};
        if (greedy_path(tokens, parents, compact) != std::vector<std::int32_t>({0, 2, 4}))
            throw std::runtime_error(
                "compact branch acceptance changed path or checked unused rows");
        compact.selection[5] = 0;
        bool bad_selection = false;
        try {
            greedy_path(tokens, parents, compact);
        } catch (const std::invalid_argument&) {
            bad_selection = true;
        }
        if (!bad_selection)
            throw std::runtime_error("compact selection accepted a consumed non-finite row");
        compact.ranks = 2;
        compact.selection = {1, 2, 1};
        if (compact.token() != 1 || compact.token(0, 1) != 2)
            throw std::runtime_error("compact draft rank order changed");
        compact.selection.clear();
        compact.logits = {-3.0F, 7.0F, 7.0F, -1.0F};
        if (compact.token() != 1 || compact.token(0, 1) != 2)
            throw std::runtime_error("host draft rank order changed");
        std::uint16_t feature_data[12]{};
        trtmc::llama::speculative::FeatureView features{feature_data, 3, 4};
        const auto last = features.slice(2, 1);
        if (last.data != feature_data + 8 || last.rows != 1 || last.width != 4)
            throw std::runtime_error("device feature row offset changed");
        bool bad_slice = false;
        try {
            features.slice(2, 2);
        } catch (const std::invalid_argument&) {
            bad_slice = true;
        }
        if (!bad_slice)
            throw std::runtime_error("out-of-bounds device feature view was accepted");
        logits[2] = 0;
        logits[3] = 5;
        if (greedy_path(tokens, parents, logits, 4) != std::vector<std::int32_t>({0}))
            throw std::runtime_error("rejected children must leave only the pending root");
        bool rejected = false;
        try {
            greedy_path({0, 1}, {-1, 1}, std::vector<float>(8), 4);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("accepted a cyclic parent relation");
        std::cout << "speculative policy tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
