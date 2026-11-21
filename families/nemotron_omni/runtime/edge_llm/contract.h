/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/task.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::nemotron_omni::edge_llm {

inline constexpr const char* kRevision = "95515c2f87fba8982db5a519f9022277667b3cc9";

/// Return whether an artifact is a normalized file below one of the two Edge roots.
inline bool safe_artifact_path(const std::string& name) {
    if (name.find('\\') != std::string::npos || name.find('\0') != std::string::npos)
        return false;
    const std::filesystem::path path(name);
    if (path.is_absolute() || path.filename().empty())
        return false;
    for (const auto& part : path)
        if (part == "." || part == "..")
            return false;
    return path.generic_string() == name &&
           (name.rfind("edge_llm/engine/", 0) == 0 || name.rfind("edge_llm/checkpoint/", 0) == 0);
}

/// Match Python platform.freedesktop_os_release precedence and missing-key fallback.
inline std::string runtime_os_version(const std::string& kernel,
                                      const std::filesystem::path& root = "/") {
    for (const auto* name : {"etc/os-release", "usr/lib/os-release"}) {
        std::ifstream release(root / name);
        if (!release)
            continue;
        std::string line, version = kernel;
        while (std::getline(release, line)) {
            if (line.rfind("VERSION_ID=", 0) != 0)
                continue;
            version = line.substr(11);
            if (!version.empty() && version.back() == '\r')
                version.pop_back();
            if (version.size() >= 2 && (version.front() == '\"' || version.front() == '\'') &&
                version.back() == version.front())
                version = version.substr(1, version.size() - 2);
        }
        return version;
    }
    return kernel;
}

/// Reject insufficient temporary storage before streaming any checkpoint sections.
inline void require_artifact_space(std::uintmax_t required, std::uintmax_t available) {
    if (available == std::numeric_limits<std::uintmax_t>::max() || available < required)
        throw std::runtime_error(
            "Nemotron Omni Edge extraction requires " + std::to_string(required) +
            " bytes; temporary storage reports " + std::to_string(available) +
            " available bytes. Set TMPDIR to a disk-backed directory with enough space.");
}

} // namespace trtmc::nemotron_omni::edge_llm
