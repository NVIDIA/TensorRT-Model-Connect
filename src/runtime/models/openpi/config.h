/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/openpi/openpi_data_plane.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace trtmc::openpi {

inline constexpr std::string_view kAuditedUpstreamCommit =
    "15a9616a00943ada6c20a0f158e3adb39df2ccac";
inline constexpr int32_t kPrefixLength = 968;
inline constexpr int32_t kMaximumPromptTokens = 200;
inline constexpr int32_t kTransformerLayers = 18;
inline constexpr int32_t kKvHeads = 1;
inline constexpr int32_t kHeadDimension = 256;
inline constexpr std::array<std::string_view, 5> kRequiredIntegritySectionNames = {
    "config.json",
    "engine_plan",
    "openpi_action_step_engine_plan",
    "tokenizer.model",
    "preprocessor_config.json",
};

struct OpenPIConfig {
    std::string profile;
    std::string precision;
    std::string parameter_dtype;
    std::string tokenizer_sha256;
    std::string normalization_sha256;
    std::string prefill_engine_sha256;
    std::string action_engine_sha256;
    int32_t action_horizon{0};
    int32_t internal_action_dim{0};
    int32_t external_action_dim{0};
    int32_t external_state_dim{0};
    int32_t prefix_length{0};
    int32_t max_token_length{0};
    int32_t num_layers{0};
    int32_t num_heads{0};
    int32_t num_kv_heads{0};
    int32_t head_dim{0};
    int32_t denoise_steps{0};
    int32_t batch_size{0};
    bool discrete_state_input{false};
    std::array<std::string, 3> camera_names;
    std::array<bool, 3> camera_mask{};
};

struct OpenPINormalization {
    QuantileStats state;
    QuantileStats actions;
};

// Parse and validate the complete model-owned runtime contract. Only the two
// explicitly audited profiles are accepted; path- or shape-based inference is
// deliberately forbidden.
OpenPIConfig parse_openpi_config(std::string_view config_json);

// Parse profile normalization statistics and require finite q01/q99 values.
// Externally visible dimensions require a positive span; padded model-only
// dimensions may have the equal zero-span statistics emitted by OpenPI.
OpenPINormalization parse_openpi_normalization(std::string_view normalization_json,
                                               const OpenPIConfig& config);

} // namespace trtmc::openpi
