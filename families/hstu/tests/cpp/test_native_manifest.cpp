/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/native_library.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace {

int failures = 0;

nlohmann::json manifest(bool dense, int major = 10, int minor = 3) {
    const auto digest = std::string(64, 'a');
    return {{"digest", digest},
            {"namespace", "trtmc_hstu_" + digest},
            {"schema_version", 2},
            {"adapter_abi", 2},
            {"provider", "original_cuda_m64"},
            {"tile_m", 64},
            {"tile_n", 128},
            {"warps", 4},
            {"attention_mode", dense ? "dense" : "paged"},
            {"creator", dense ? "HstuDenseAttention" : "HstuPagedAttention"},
            {"creator_version", "1"},
            {"dtype", "bf16"},
            {"heads", 4},
            {"head_dim", 64},
            {"page_size", dense ? 0 : 128},
            {"max_capacity", 1024},
            {"scaling_seqlen", 1024},
            {"causal", true},
            {"target_group_size", 1},
            {"context", false},
            {"gpu_arch", "sm_" + std::to_string(major) + std::to_string(minor)},
            {"compute_capability", {major, minor}}};
}

void check(const nlohmann::json& value, bool cache, int major, int minor, bool expected,
           const char* label) {
    bool accepted = false;
    try {
        const auto text = value.dump();
        trtmc::hstu::validate_native_attention_manifest({text.begin(), text.end()}, cache, major,
                                                        minor);
        accepted = true;
    } catch (const std::exception&) {
    }
    if (accepted != expected) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

void supported_targets() {
    for (const auto& [major, minor] :
         {std::pair{8, 0}, {8, 6}, {9, 0}, {10, 0}, {10, 3}, {12, 0}}) {
        for (const bool dense : {false, true}) {
            const auto value = manifest(dense, major, minor);
            check(value, !dense, major, minor, true,
                  "same provider contract for each build target");
            check(value, !dense, major, (minor + 1) % 10, false,
                  "reject a different serving target");
            check(value, dense, major, minor, false,
                  "dense/paged mode must match cache configuration");
        }
    }
}

void reject_changed_contract() {
    const nlohmann::json changes = {{"schema_version", 3},
                                    {"adapter_abi", 1},
                                    {"provider", "unqualified"},
                                    {"tile_m", 128},
                                    {"tile_n", 64},
                                    {"warps", 8},
                                    {"digest", "a"},
                                    {"namespace", "trtmc_hstu_other"},
                                    {"creator", "HstuDenseAttention"},
                                    {"creator_version", "2"},
                                    {"dtype", "fp16"},
                                    {"heads", 8},
                                    {"head_dim", 128},
                                    {"page_size", 64},
                                    {"max_capacity", 2048},
                                    {"scaling_seqlen", 512},
                                    {"causal", false},
                                    {"target_group_size", 2},
                                    {"context", true},
                                    {"gpu_arch", "sm_90"}};
    for (auto item = changes.begin(); item != changes.end(); ++item) {
        auto value = manifest(false);
        value[item.key()] = item.value();
        check(value, true, 10, 3, false, item.key().c_str());
    }
    for (const auto& target : nlohmann::json::array({nullptr,
                                                     "10,3",
                                                     {10},
                                                     {10, 3, 0},
                                                     {10.0, 3},
                                                     {10, -1},
                                                     {7, 0},
                                                     {100, 0},
                                                     {10, 13},
                                                     {4294967306LL, 3}})) {
        auto value = manifest(false);
        value["compute_capability"] = target;
        check(value, true, 10, 3, false, "reject malformed target without integer truncation");
    }
    for (const auto* field :
         {"provider", "tile_m", "compute_capability", "adapter_abi", "attention_mode"}) {
        auto value = manifest(false);
        value.erase(field);
        check(value, true, 10, 3, false, "reject missing contract fields");
    }
}

void legacy_contract() {
    for (const bool dense : {false, true}) {
        auto value = manifest(dense);
        value["schema_version"] = 1;
        value["adapter_abi"] = 1;
        value["gpu_arch"] = "sm_103a";
        for (const auto* field : {"provider", "tile_m", "tile_n", "warps"})
            value.erase(field);
        if (!dense)
            value.erase("attention_mode");
        check(value, !dense, 10, 3, true, "preserve original dense/paged schema1 artifacts");
        check(value, !dense, 9, 0, false, "legacy Blackwell artifact cannot run as Hopper");
        value["compute_capability"] = {9, 0};
        value["gpu_arch"] = "sm_90";
        check(value, !dense, 9, 0, false, "legacy schema cannot claim a generic target");
    }
}

} // namespace

int main() {
    supported_targets();
    reject_changed_contract();
    legacy_contract();
    return failures == 0 ? 0 : 1;
}
