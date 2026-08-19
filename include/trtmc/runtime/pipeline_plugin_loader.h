/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace trtmc {

struct ModelPluginInfo {
    const char* model_id;
    const char* runtime_strategy;
    const char* library_name;
};

// Generated from python/tensorrt_model_connect/models/*/MODEL.toml at configure time.
const std::vector<ModelPluginInfo>& runtime_model_plugin_index();
std::optional<std::string> model_plugin_id_for_strategy(const std::string& strategy);
std::string model_plugin_library_name(const std::string& model_id);

// Load the model plugin that owns strategy. Search paths are directories that
// contain libtrtmc_model_<model>.so; TRTMC_MODEL_PLUGIN_DIR and build/install
// defaults are consulted after these explicit paths. When
// TRTMC_MODEL_PLUGIN_STRICT=1, only explicit paths and TRTMC_MODEL_PLUGIN_DIR
// are used, so an installed or stale build-tree DSO cannot satisfy a CI proof.
void load_model_plugin_for_strategy(const std::string& strategy,
                                    const std::vector<std::string>& search_paths = {});

} // namespace trtmc
