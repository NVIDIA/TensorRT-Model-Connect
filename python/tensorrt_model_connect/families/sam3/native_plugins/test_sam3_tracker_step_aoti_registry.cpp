/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <tvm/ffi/c_api.h>

extern "C" int
trtmc_sam3_tracker_step_register_pipeline(const char* global_name, const char* encoder_path,
                                          const char* decoder_path, const char* encoder_sha256,
                                          const char* decoder_sha256, int32_t batch_size) noexcept;

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
          "resolve registered tracker-step pipeline global");
    return function;
}

} // namespace

int main() {
    constexpr const char* kName = "trtmc.sam3.tracker_step.b1.split_aoti.0123456789abcdef0123";
    constexpr const char* kEncoderHash =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    constexpr const char* kDecoderHash =
        "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0";
    constexpr const char* kOtherHash =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    check(trtmc_sam3_tracker_step_register_pipeline(kName, "/first/encoder.pt2",
                                                    "/first/decoder.pt2", kEncoderHash,
                                                    kDecoderHash, 1) == 0,
          "register initial tracker-step pipeline generation");
    TVMFFIObjectHandle first = resolve(kName);
    check(trtmc_sam3_tracker_step_register_pipeline(kName, "/relocated/encoder.pt2",
                                                    "/relocated/decoder.pt2", kEncoderHash,
                                                    kDecoderHash, 1) == 0,
          "register relocated identical tracker-step pipeline generation");
    TVMFFIObjectHandle second = resolve(kName);
    check(first != second, "relocated tracker-step pipeline publishes a new function generation");
    check(trtmc_sam3_tracker_step_register_pipeline(kName, "/conflict/encoder.pt2",
                                                    "/conflict/decoder.pt2", kOtherHash,
                                                    kDecoderHash, 1) == -2,
          "reject conflicting tracker-step hashes for an existing global");
    check(trtmc_sam3_tracker_step_register_pipeline(kName, "/wrong/encoder.pt2",
                                                    "/wrong/decoder.pt2", kEncoderHash,
                                                    kDecoderHash, 2) == -1,
          "reject tracker-step batch/global disagreement");
    check(trtmc_sam3_tracker_step_register_pipeline(kName, "/bad/encoder.pt2", "/bad/decoder.pt2",
                                                    "not-a-hash", kDecoderHash, 1) == -1,
          "reject non-content-addressed tracker-step hash");

    // The bridge retains both Entry generations for process lifetime, so each
    // TensorRT context may keep its independently resolved function handle.
    TVMFFIObjectDecRef(first);
    TVMFFIObjectDecRef(second);
    std::cout << "PASS: SAM3 tracker-step registration generations\n";
    return 0;
}
