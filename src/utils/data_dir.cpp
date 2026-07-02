/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/data_dir.h"

#include <string>

#ifndef TRTMC_SOURCE_DIR
#define TRTMC_SOURCE_DIR "."
#endif

namespace trtmc {

// Process-wide runtime source-dir setting. Populated by
// set_source_dir(...) from pipeline_factory after resolving the
// platform.* registry namespace. Empty (the default) means "use the
// compile-time TRTMC_SOURCE_DIR." Replaces the old TRTMC_DATA_DIR env var.
static std::string& mutable_source_dir() {
    static std::string value;
    return value;
}

void set_source_dir(const std::string& value) {
    mutable_source_dir() = value;
}

std::string source_dir() {
    const auto& configured_value = mutable_source_dir();
    if (!configured_value.empty())
        return configured_value;
    return TRTMC_SOURCE_DIR;
}

std::string scripts_dir() {
    return source_dir() + "/scripts";
}

std::string models_dir() {
    return source_dir() + "/models";
}

std::string script_path(const char* script_name) {
    return scripts_dir() + "/" + script_name;
}

std::string model_path(const char* relative_path) {
    return models_dir() + "/" + relative_path;
}

} // namespace trtmc
