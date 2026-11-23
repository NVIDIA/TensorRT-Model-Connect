/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/pika_h3_flex/runtime/math.h"

#include <stdexcept>
#include <string>

#if TRTMC_PIKA_HAS_MATH
#include "trtmc/byok.h"

#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#endif

namespace trtmc::pika_h3_flex {
void initialize_math() {
#if TRTMC_PIKA_HAS_MATH
    static std::once_flag once;
    std::call_once(once, [] {
        Dl_info info{};
        if (!dladdr(reinterpret_cast<const void*>(&trtmc_pika_h3_flex_initialize_math), &info) ||
            !info.dli_fname)
            throw std::runtime_error("Pika H3 Flex could not locate its mathematical bindings");
        const auto library = std::filesystem::canonical(info.dli_fname).string();
        for (const auto* operation :
             {"linear", "linear_bias", "norm", "silu", "mm", "sdpa", "time_features"})
            load_byok_kernel(library, std::string("pika_h3_flex_") + operation,
                             std::string("pika_h3_flex.") + operation + "_v1");
    });
#else
    throw std::runtime_error("Pika H3 Flex transformer math requires the TVM-FFI bridge");
#endif
}
} // namespace trtmc::pika_h3_flex

extern "C" const char* trtmc_pika_h3_flex_initialize_math() noexcept {
    static thread_local std::string message;
    try {
        trtmc::pika_h3_flex::initialize_math();
        return nullptr;
    } catch (const std::exception& error) {
        try {
            message = error.what();
            return message.c_str();
        } catch (...) {
            return "Pika H3 Flex mathematical binding initialization failed";
        }
    } catch (...) {
        return "Pika H3 Flex mathematical binding initialization failed";
    }
}
