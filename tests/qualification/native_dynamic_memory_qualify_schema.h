/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace trtmc::qualification {

inline nlohmann::json make_sequential_request_samples() {
    return nlohmann::json::array();
}

inline nlohmann::json make_runtime_phase_memory_samples() {
    return nlohmann::json::array();
}

inline nlohmann::json make_runtime_phase_memory_sample(std::string phase, std::uint32_t device,
                                                       std::uint64_t free_bytes,
                                                       std::uint64_t total_bytes,
                                                       std::uint64_t process_used_bytes) {
    return {
        {"phase", std::move(phase)},
        {"device", device},
        {"free_bytes", free_bytes},
        {"total_bytes", total_bytes},
        {"used_bytes", total_bytes - free_bytes},
        {"process_used_bytes", process_used_bytes},
    };
}

inline void attach_runtime_phase_memory_samples(nlohmann::json& lifetime,
                                                const nlohmann::json& samples) {
    lifetime["runtime_phase_memory_samples"] = samples;
}

} // namespace trtmc::qualification
