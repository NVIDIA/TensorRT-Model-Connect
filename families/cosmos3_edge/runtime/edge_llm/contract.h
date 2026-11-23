/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/task.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::cosmos3_edge::edge_llm {

#ifdef TRTMC_COSMOS3_EDGE_012
inline constexpr const char* kRevision = "513dcd18291e587c80928911a480761352f3c2d8";
#else
inline constexpr const char* kRevision = "95515c2f87fba8982db5a519f9022277667b3cc9";
#endif

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

} // namespace trtmc::cosmos3_edge::edge_llm
