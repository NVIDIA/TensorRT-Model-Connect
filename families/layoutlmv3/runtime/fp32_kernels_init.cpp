/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/layoutlmv3/runtime/fp32_kernels.h"

#include <new>
#include <stdexcept>
#include <string>

#if TRTMC_LAYOUTLMV3_HAS_FP32_KERNELS
#include "trtmc/byok.h"

#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#endif

namespace trtmc::layoutlmv3 {
void initialize_fp32_kernels() {
#if TRTMC_LAYOUTLMV3_HAS_FP32_KERNELS
    static std::once_flag once;
    std::call_once(once, [] {
        Dl_info information{};
        if (!dladdr(reinterpret_cast<const void*>(&trtmc_layoutlmv3_initialize_fp32_kernels),
                    &information) ||
            !information.dli_fname)
            throw std::runtime_error("LAYOUTLMV3 could not locate its native kernel library");
        const auto library = std::filesystem::canonical(information.dli_fname).string();
        for (const auto* operation : {"norm", "linear", "softmax", "gelu"})
            load_byok_kernel(library, std::string("layoutlmv3_") + operation + "_fp32",
                             std::string("layoutlmv3.") + operation + "_cuda_fp32_v1");
    });
#else
    throw std::runtime_error("LAYOUTLMV3 FP32 vision kernels require the TVM-FFI bridge");
#endif
}
} // namespace trtmc::layoutlmv3

extern "C" const char* trtmc_layoutlmv3_initialize_fp32_kernels() noexcept {
    static thread_local std::string error;
    try {
        trtmc::layoutlmv3::initialize_fp32_kernels();
        return nullptr;
    } catch (const std::bad_alloc&) {
        return "LAYOUTLMV3 kernel initialization ran out of memory";
    } catch (const std::exception& exception) {
        try {
            error = exception.what();
            return error.c_str();
        } catch (...) {
            return "LAYOUTLMV3 kernel initialization ran out of memory";
        }
    } catch (...) {
        return "LAYOUTLMV3 kernel initialization failed";
    }
}
