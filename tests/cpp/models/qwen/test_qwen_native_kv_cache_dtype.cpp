/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen/plugin_helpers.h"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        ++failures;
    }
}

void expect_dtype(const std::string& config_json, const std::string& precision,
                  trtmc::DType expected, const char* name) {
    try {
        check(trtmc::resolve_qwen_native_kv_cache_dtype(config_json, precision) == expected, name);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s threw unexpectedly: %s\n", name, error.what());
        ++failures;
    }
}

void expect_failure(const std::string& config_json, const char* name) {
    try {
        (void)trtmc::resolve_qwen_native_kv_cache_dtype(config_json, "fp16");
        std::fprintf(stderr, "FAIL: %s did not throw\n", name);
        ++failures;
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find("native_kv_cache_dtype") != std::string::npos, name);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s threw the wrong exception: %s\n", name, error.what());
        ++failures;
    }
}

} // namespace

int main() {
    const std::string contract_v1 = R"({"native_kv_contract_version":1})";
    expect_dtype(contract_v1, "fp16", trtmc::DType::kFloat16, "v1 inherits FP16");
    expect_dtype(contract_v1, "bf16", trtmc::DType::kBFloat16, "v1 inherits BF16");

    expect_dtype(R"({"native_kv_contract_version":2,"native_kv_cache_dtype":"fp16"})", "bf16",
                 trtmc::DType::kFloat16, "v2 selects explicit FP16");
    expect_dtype(R"({"native_kv_contract_version":2,"native_kv_cache_dtype":"bf16"})", "fp16",
                 trtmc::DType::kBFloat16, "v2 selects explicit BF16");

    expect_failure(R"({"native_kv_contract_version":2})", "v2 rejects missing dtype");
    expect_failure(R"({"native_kv_contract_version":2,"native_kv_cache_dtype":"fp32"})",
                   "v2 rejects invalid dtype");

    if (failures == 0)
        std::fprintf(stderr, "All Qwen native KV cache dtype tests passed.\n");
    return failures;
}
