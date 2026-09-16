/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstring>

extern "C" {
int trtmc_test_byok_calls = 0;
}

extern "C" const char* trtmc_load_byok_kernel(const char* library, const char* function,
                                              const char* kernel_name) noexcept {
    static thread_local char error[256];
    ++trtmc_test_byok_calls;
    if (std::strcmp(function, "run") != 0) {
        std::snprintf(error, sizeof(error), "unknown fixture function: %s", function);
        return error;
    }
    if (std::strcmp(kernel_name, "fixture.copy") != 0)
        return "unknown fixture kernel name";
    std::FILE* input = std::fopen(library, "rb");
    if (!input)
        return "fixture module file is missing";
    std::fclose(input);
    return nullptr;
}
