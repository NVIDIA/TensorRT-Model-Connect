/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/phi_moe/runtime/eos_tokens.h"

#include <iostream>

int main() {
    int failures = 0;
    const auto check = [&](bool condition, const char* label) {
        if (!condition) {
            std::cerr << "FAIL: " << label << '\n';
            ++failures;
        }
    };
    // Phi-tiny-MoE-instruct revision 2fe50e88d0e2a5a132563815686ea0dcc8e252b5.
    const auto ids = trtmc::phi_moe_parse_eos_tokens(nlohmann::json::parse("[32000,32001,32007]"));
    check(ids == std::vector<int32_t>({32000, 32001, 32007}), "all checkpoint EOS ids survive");
    for (int32_t id : ids)
        check(trtmc::phi_moe_is_eos(id, ids.front(), ids), "checkpoint stop token is recognized");
    check(!trtmc::phi_moe_is_eos(13, ids.front(), ids), "ordinary newline continues");
    check(!trtmc::phi_moe_is_eos(32007, ids.front(), ids, 17),
          "explicit override replaces defaults");
    check(trtmc::phi_moe_is_eos(17, ids.front(), ids, 17), "explicit override is honored");
    check(trtmc::phi_moe_parse_eos_tokens(17) == std::vector<int32_t>({17}), "scalar bundle loads");
    check(trtmc::phi_moe_is_eos(17, 17, {}), "legacy config scalar fallback");
    for (const auto& invalid : {nlohmann::json::parse("[]"), nlohmann::json::parse("[32000,null]"),
                                nlohmann::json::parse("null")}) {
        bool rejected = false;
        try {
            (void)trtmc::phi_moe_parse_eos_tokens(invalid);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "invalid checkpoint EOS metadata is rejected");
    }
    return failures == 0 ? 0 : 1;
}
