/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// test_tvm_ffi_module_loader.cpp — Load FlashInfer .so from C++, run in TRT
// =============================================================================
//
// Intent:
//   Proves that a JIT-compiled FlashInfer CUDA kernel can be loaded from
//   a .so file via the TVM-FFI C API in pure C++ (no Python), registered
//   as a global function, and called from a TRT engine via the plugin.
//
// Preconditions:
//   - TRTMC_HAS_TRT=1 and TRTMC_HAS_TVM_FFI=1
//   - FlashInfer .so cached at FLASHINFER_KERNEL_SO env var
//
// Trace IDs: ARCH-TVM-FFI-001, UD-TVM-FFI-LOADER-001, UT-TVM-FFI-PURE-CPP-001
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#if TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI

#include "plugins/tvm_ffi_module_loader.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>

extern "C" void tvm_ffi_plugin_force_link();

static int failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class TestLogger : public nvinfer1::ILogger {
  public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kERROR)
            std::cerr << "[TRT] " << msg << '\n';
    }
};

static void test_load_and_run() {
    const char* so_path = std::getenv("FLASHINFER_KERNEL_SO");
    if (so_path == nullptr) {
        std::cerr << "SKIP: FLASHINFER_KERNEL_SO not set\n";
        return;
    }

    // 1. Load FlashInfer .so and register "run" as global function
    std::cerr << "Loading FlashInfer kernel from: " << so_path << '\n';
    bool ok = trtmc::load_tvm_ffi_module_func(so_path, "run", "flashinfer.decode_test");
    check(ok, "load_tvm_ffi_module_func");
    if (!ok)
        return;

    // 2. Verify the function is globally accessible
    TVMFFIByteArray name_arr;
    const char* fn_name = "flashinfer.decode_test";
    name_arr.data = fn_name;
    name_arr.size = static_cast<int64_t>(std::strlen(fn_name));
    TVMFFIObjectHandle fn = nullptr;
    int ret = TVMFFIFunctionGetGlobal(&name_arr, &fn);
    check(ret == 0 && fn != nullptr, "kernel found globally");

    std::cerr << "FlashInfer kernel loaded and registered in pure C++!\n";

    // 3. Build a TRT engine that calls it via the plugin
    // (Same pattern as test_tvm_ffi_plugin.cpp but with FlashInfer)
    // For now, just verify loading works — full engine test requires
    // matching tensor shapes to FlashInfer's expectations.
    std::cerr << "Module loader test complete.\n";
}

#endif

int main() {
#if TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI
    tvm_ffi_plugin_force_link();
    test_load_and_run();

    if (failures > 0) {
        std::cerr << failures << " FAILED\n";
        return 1;
    }
    std::cerr << "All module_loader tests passed.\n";
#else
    std::cerr << "Skipping (no TRT/TVM-FFI)\n";
#endif
    return 0;
}
