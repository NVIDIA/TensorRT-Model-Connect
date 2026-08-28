/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/registry/pipeline_plugin_loader_internal.h"

#include <iostream>
#include <string>
#include <vector>

#ifndef TRTMC_TEST_MODEL_PLUGIN_MISSING_ABI_DSO
#error "TRTMC_TEST_MODEL_PLUGIN_MISSING_ABI_DSO must be defined"
#endif

#ifndef TRTMC_TEST_MODEL_PLUGIN_WRONG_ABI_DSO
#error "TRTMC_TEST_MODEL_PLUGIN_WRONG_ABI_DSO must be defined"
#endif

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

void expect_abi_rejection(const std::string& path, const std::string& expected_error,
                          const char* description) {
    void* handle = nullptr;
    trtmc::detail::RegisterModelPluginFn register_fn = nullptr;
    std::vector<std::string> errors;

    const bool opened = trtmc::detail::open_model_plugin_entrypoints(path, "synthetic-model",
                                                                     handle, register_fn, errors);

    check(!opened, description);
    check(handle == nullptr, "rejected model plugin handle is closed");
    check(register_fn == nullptr, "rejected model plugin register entrypoint is cleared");
    check(errors.size() == 1, "ABI rejection reports one load error");
    check(!errors.empty() && errors.front().find(expected_error) != std::string::npos,
          "ABI rejection reports the expected reason");
}

} // namespace

int main() {
    expect_abi_rejection(TRTMC_TEST_MODEL_PLUGIN_MISSING_ABI_DSO,
                         "missing trtmc_model_plugin_abi_version",
                         "missing model plugin ABI is rejected");
    expect_abi_rejection(TRTMC_TEST_MODEL_PLUGIN_WRONG_ABI_DSO,
                         "model plugin ABI mismatch, expected 2 but got 1",
                         "wrong model plugin ABI is rejected");

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }
    std::cerr << "All model_plugin_abi tests passed" << std::endl;
    return 0;
}
