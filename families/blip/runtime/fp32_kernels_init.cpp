/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/blip/runtime/fp32_kernels.h"

#include <new>
#include <stdexcept>
#include <string>

#if TRTMC_BLIP_HAS_FP32_KERNELS
#include "trtmc/byok.h"

#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#endif

namespace trtmc::blip {
void initialize_fp32_kernels() {
#if TRTMC_BLIP_HAS_FP32_KERNELS
    static std::once_flag once;
    std::call_once(once, [] {
        Dl_info information{};
        if (!dladdr(reinterpret_cast<const void*>(&trtmc_blip_initialize_fp32_kernels),
                    &information) ||
            !information.dli_fname)
            throw std::runtime_error("BLIP could not locate its native kernel library");
        const auto library = std::filesystem::canonical(information.dli_fname).string();
        for (const auto* operation : {"norm", "linear", "softmax", "gelu"})
            load_byok_kernel(library, std::string("blip_") + operation + "_fp32",
                             std::string("blip.vision.") + operation + "_fp32");
    });
#else
    throw std::runtime_error("BLIP FP32 vision kernels require the TVM-FFI bridge");
#endif
}
} // namespace trtmc::blip

extern "C" const char* trtmc_blip_initialize_fp32_kernels() noexcept {
    static thread_local std::string error;
    try {
        trtmc::blip::initialize_fp32_kernels();
        return nullptr;
    } catch (const std::bad_alloc&) {
        return "BLIP kernel initialization ran out of memory";
    } catch (const std::exception& exception) {
        try {
            error = exception.what();
            return error.c_str();
        } catch (...) {
            return "BLIP kernel initialization ran out of memory";
        }
    } catch (...) {
        return "BLIP kernel initialization failed";
    }
}
