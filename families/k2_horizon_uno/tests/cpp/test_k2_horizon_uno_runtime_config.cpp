/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/k2_horizon_uno/runtime/runtime_config.h"

#include <iostream>
#include <string_view>

namespace {

int failures = 0;

bool accepts(std::string_view json) {
    try {
        trtmc::k2_horizon_uno::validate_runtime_config_json(json);
        return true;
    } catch (...) {
        return false;
    }
}

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    check(accepts(R"({"max_cache_length":256})"), "valid cache length is accepted");
    check(!accepts(R"({})"), "missing cache length is rejected");
    check(!accepts(R"({"max_cache_length":"256"})"), "wrong type is rejected");
    check(!accepts(R"({"max_cache_length":256.9})"), "fractional cache length is rejected");
    check(!accepts(R"({"max_cache_length":7})"), "cache shorter than one block is rejected");
    check(!accepts(R"({"max_cache_length":524289})"), "model position limit is enforced");
    check(!accepts(R"({"max_cache_length":256,"extra":true})"), "extra fields are rejected");
    check(!accepts("{"), "invalid JSON is rejected");
    return failures == 0 ? 0 : 1;
}
