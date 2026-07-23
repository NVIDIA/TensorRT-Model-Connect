/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/openpi/api.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::openpi::tool {

struct ActionCameraFile {
    std::string name;
    std::string path;
    bool valid{true};
};

struct ActionRequestFile {
    std::string prompt;
    std::array<ActionCameraFile, 3> cameras;
    std::vector<float> state;
    std::vector<float> initial_noise;
    int32_t seed{-1};
    int32_t denoise_steps{-1};
};

// Parse the OpenPI runner's `--request-json` schema. The parser rejects
// duplicate, unknown, missing, mistyped, non-finite, and out-of-range values.
ActionRequestFile parse_action_request_json(std::string_view text);
ActionRequestFile read_action_request_json(const std::string& path);

// Serialize the public action result schema after validating the typed result's
// shape and numeric values. Actions use row-major [horizon, action_dim] order.
std::string serialize_action_result_json(const ActionResult& result);

} // namespace trtmc::openpi::tool
