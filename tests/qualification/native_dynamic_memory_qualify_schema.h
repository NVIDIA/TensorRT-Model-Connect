/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
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
                                                       nlohmann::json sample) {
    if (!sample.is_object())
        throw std::invalid_argument("runtime phase memory sample must be an object");
    sample["phase"] = std::move(phase);
    sample["device"] = device;
    return sample;
}

inline void attach_runtime_phase_memory_samples(nlohmann::json& lifetime,
                                                const nlohmann::json& samples) {
    lifetime["runtime_phase_memory_samples"] = samples;
}

} // namespace trtmc::qualification
