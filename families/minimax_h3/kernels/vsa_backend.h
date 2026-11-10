/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>

namespace trtmc::minimax_h3 {

enum class VsaBackend { kGeneric, kBlackwell };

inline const char* vsa_backend_name(VsaBackend backend) noexcept {
    return backend == VsaBackend::kBlackwell ? "blackwell" : "generic";
}

inline bool is_optimized_blackwell(int device_major, int device_minor) noexcept {
    return device_major == 10 && (device_minor == 0 || device_minor == 3);
}

// The Blackwell kernels run wherever this library was compiled with them for the active GPU.
inline VsaBackend select_vsa_backend(int device_major, int device_minor,
                                     bool has_blackwell_kernels) {
    if (device_major < 8)
        throw std::runtime_error("FastH3 VSA requires a CUDA GPU with compute capability 8.0+");
    return has_blackwell_kernels && is_optimized_blackwell(device_major, device_minor)
               ? VsaBackend::kBlackwell
               : VsaBackend::kGeneric;
}

} // namespace trtmc::minimax_h3
