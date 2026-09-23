/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>
#include <string_view>

namespace trtmc::minimax_h3 {

enum class VsaBackend { kGeneric, kBlackwell };

inline const char* vsa_backend_name(VsaBackend backend) noexcept {
    return backend == VsaBackend::kBlackwell ? "blackwell" : "generic";
}

inline bool is_optimized_blackwell(int device_major, int device_minor) noexcept {
    return device_major == 10 && (device_minor == 0 || device_minor == 3);
}

inline VsaBackend select_vsa_backend(std::string_view request, int device_major, int device_minor,
                                     bool has_blackwell_kernels) {
    if (request != "auto" && request != "generic" && request != "blackwell")
        throw std::invalid_argument(
            "TRTMC_MINIMAX_H3_VSA_BACKEND must be auto, generic, or blackwell");
    if (device_major < 8)
        throw std::runtime_error("FastH3 VSA requires a CUDA GPU with compute capability 8.0+");

    const bool optimized =
        has_blackwell_kernels && is_optimized_blackwell(device_major, device_minor);
    if (request == "blackwell" && !optimized)
        throw std::runtime_error(
            "FastH3 VSA Blackwell backend was requested, but this library has no optimized "
            "kernel for the active GPU; use auto or generic, or rebuild for its architecture");
    if (request == "generic")
        return VsaBackend::kGeneric;
    return optimized ? VsaBackend::kBlackwell : VsaBackend::kGeneric;
}

} // namespace trtmc::minimax_h3
