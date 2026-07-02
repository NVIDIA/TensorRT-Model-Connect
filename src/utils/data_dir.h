/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace trtmc {

// Returns the project source directory. Resolution:
// 1. platform.source_dir registry value (if non-empty and set via
//    set_source_dir from pipeline_factory)
// 2. TRTMC_SOURCE_DIR compile-time define
std::string source_dir();

// Set the runtime runtime source-dir setting. Called by pipeline_factory after
// resolving the platform.* registry namespace. Replaces the old
// TRTMC_DATA_DIR env var.
void set_source_dir(const std::string& value);

// Returns path to scripts/ directory.
std::string scripts_dir();

// Returns path to models/ directory.
std::string models_dir();

// Resolves a script by name under scripts_dir().
std::string script_path(const char* script_name);

// Resolves a model directory under models_dir().
std::string model_path(const char* relative_path);

} // namespace trtmc
