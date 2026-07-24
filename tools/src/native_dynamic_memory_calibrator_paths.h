/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace trtmc::qualification {

struct InternalCalibratorSearchPaths {
    std::vector<std::string> backend;
    std::vector<std::string> model_plugin;
};

inline void append_unique_path(std::vector<std::filesystem::path>& paths,
                               const std::filesystem::path& candidate) {
    const auto normalized = candidate.lexically_normal();
    for (const auto& existing : paths) {
        if (existing == normalized)
            return;
    }
    paths.push_back(normalized);
}

inline void append_python_package_bins(const std::filesystem::path& prefix,
                                       std::vector<std::filesystem::path>& paths) {
    std::error_code error;
    for (const auto& library_dir : {prefix / "lib", prefix / "lib64"}) {
        if (!std::filesystem::is_directory(library_dir, error)) {
            error.clear();
            continue;
        }
        for (std::filesystem::directory_iterator iterator(library_dir, error), end;
             !error && iterator != end; iterator.increment(error)) {
            if (!iterator->is_directory(error)) {
                error.clear();
                continue;
            }
            const auto name = iterator->path().filename().string();
            if (name.rfind("python", 0) != 0)
                continue;
            const auto candidate =
                iterator->path() / "site-packages/tensorrt_model_connect/bin";
            if (std::filesystem::is_directory(candidate, error))
                append_unique_path(paths, candidate);
            error.clear();
        }
        error.clear();
    }
}

inline InternalCalibratorSearchPaths internal_calibrator_search_paths(
    const std::filesystem::path& executable,
    const std::filesystem::path& installed_runtime_library_relative) {
    const auto absolute =
        std::filesystem::absolute(executable).lexically_normal();
    const auto helper_dir = absolute.parent_path();
    const auto companion_dir = helper_dir.parent_path();
    const auto installed_library_dir =
        (helper_dir / installed_runtime_library_relative).lexically_normal();

    std::vector<std::filesystem::path> common{
        helper_dir,
        companion_dir,
        installed_library_dir,
    };

    // A wheel installs the public native CLI under <venv>/bin and its private
    // helper under <venv>/bin/.trtmc-internal. Backend and model DSOs remain in
    // <venv>/lib/pythonX.Y/site-packages/tensorrt_model_connect/bin. Resolve
    // that package directory from the wheel prefix instead of duplicating DSOs.
    if (helper_dir.filename() == ".trtmc-internal" &&
        companion_dir.filename() == "bin") {
        append_python_package_bins(companion_dir.parent_path(), common);
    }

    std::vector<std::filesystem::path> model = common;
    append_unique_path(model, helper_dir / "models/qwen");
    append_unique_path(model, helper_dir / "models/llama");
    append_unique_path(model, installed_library_dir / "trtmc/models/qwen");
    append_unique_path(model, installed_library_dir / "trtmc/models/llama");

    InternalCalibratorSearchPaths result;
    for (const auto& path : common)
        result.backend.push_back(path.string());
    for (const auto& path : model)
        result.model_plugin.push_back(path.string());
    return result;
}

} // namespace trtmc::qualification
