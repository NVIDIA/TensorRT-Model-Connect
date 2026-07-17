/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <tvm/ffi/c_api.h>

extern "C" int trtmc_sam3_tracker_memory_register_package(const char* global_name,
                                                          const char* package_path,
                                                          const char* package_sha256,
                                                          const char* policy,
                                                          int32_t batch_size) noexcept;

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

TVMFFIObjectHandle resolve(const char* global_name) {
    const TVMFFIByteArray name{global_name, std::strlen(global_name)};
    TVMFFIObjectHandle function = nullptr;
    check(TVMFFIFunctionGetGlobal(&name, &function) == 0 && function != nullptr,
          "resolve registered memory package global");
    return function;
}

} // namespace

int main() {
    constexpr const char* kName = "trtmc.sam3.tracker_memory.soft.b1.fixed.0123456789abcdef0123";
    constexpr const char* kHash =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    constexpr const char* kOtherHash =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    check(trtmc_sam3_tracker_memory_register_package(kName, "/first/package.pt2", kHash, "soft",
                                                     1) == 0,
          "register initial memory package generation");
    TVMFFIObjectHandle first = resolve(kName);
    check(trtmc_sam3_tracker_memory_register_package(kName, "/relocated/package.pt2", kHash, "soft",
                                                     1) == 0,
          "register relocated identical memory package generation");
    TVMFFIObjectHandle second = resolve(kName);
    check(first != second, "relocated package publishes a new global function generation");
    check(trtmc_sam3_tracker_memory_register_package(kName, "/conflict/package.pt2", kOtherHash,
                                                     "soft", 1) == -2,
          "reject conflicting package hash for an existing global");
    check(trtmc_sam3_tracker_memory_register_package(kName, "/wrong/policy.pt2", kHash, "hard",
                                                     1) == -1,
          "reject policy/global disagreement");
    check(trtmc_sam3_tracker_memory_register_package(kName, "/wrong/batch.pt2", kHash, "soft", 2) ==
              -1,
          "reject batch/global disagreement");
    check(trtmc_sam3_tracker_memory_register_package(kName, "/bad/hash.pt2", "not-a-hash", "soft",
                                                     1) == -1,
          "reject non-content-addressed package hash");

    // Both references remain valid after global replacement because the
    // bridge retains every Entry generation until process exit.
    TVMFFIObjectDecRef(first);
    TVMFFIObjectDecRef(second);
    std::cout << "PASS: SAM3 tracker-memory registration generations\n";
    return 0;
}
