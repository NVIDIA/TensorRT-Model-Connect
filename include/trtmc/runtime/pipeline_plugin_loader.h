/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace trtmc {

// Bump whenever the C++ model-plugin boundary changes incompatibly. Every
// generated model DSO exports this value so the loader can reject stale or
// mixed-install plugins before calling into them.
inline constexpr std::uint32_t kTrtmcModelPluginApiAbiVersion = 1U;

struct ModelPluginInfo {
    const char* model_id;
    const char* runtime_strategy;
    const char* library_name;
};

struct LegacyStrategyAlias {
    const char* model_id;
    const char* legacy_strategy;
    const char* match_op;
    const char* config_key;
    const char* match_value;
    const char* target_strategy;
};

// Generated from src/runtime/models/*/MODEL.toml at configure time.
const std::vector<ModelPluginInfo>& runtime_model_plugin_index();
const std::vector<LegacyStrategyAlias>& legacy_runtime_strategy_alias_index();

std::optional<std::string> default_runtime_strategy();
std::optional<std::string> model_plugin_id_for_strategy(const std::string& strategy);
std::string model_plugin_library_name(const std::string& model_id);
std::optional<std::string> legacy_runtime_strategy_alias_target(const std::string& strategy,
                                                                const std::string& config_text);

// Load the model plugin that owns strategy. Search paths are directories that
// contain libtrtmc_model_<model>.so; TRTMC_MODEL_PLUGIN_DIR and build/install
// defaults are consulted after these explicit paths. When
// TRTMC_MODEL_PLUGIN_STRICT=1, only explicit paths and TRTMC_MODEL_PLUGIN_DIR
// are used, so an installed or stale build-tree DSO cannot satisfy a CI proof.
void load_model_plugin_for_strategy(const std::string& strategy,
                                    const std::vector<std::string>& search_paths = {});

} // namespace trtmc
