/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>
#include <string>

extern "C" const char* trtmc_load_byok_kernel(const char* library, const char* function,
                                              const char* kernel_name) noexcept;

namespace trtmc {

// Load one TVM-FFI module function and publish it under the exact name stored
// in a TensorRT BYOK plugin. The module remains loaded for the process lifetime.
inline void load_byok_kernel(const std::string& library, const std::string& function,
                             const std::string& kernel_name) {
    if (const char* error =
            trtmc_load_byok_kernel(library.c_str(), function.c_str(), kernel_name.c_str())) {
        throw std::runtime_error(error);
    }
}

} // namespace trtmc
