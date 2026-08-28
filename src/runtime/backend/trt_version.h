/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// TensorRT version helpers used by the TensorRT-free core runtime.
// This file must not include TensorRT headers; runtime probing is done through
// the platform shared-library boundary in trt_version.cpp.

#include <optional>
#include <string>
#include <vector>

namespace trtmc {

struct TrtVersion {
    int major{-1};
    int minor{-1};
    int patch{-1};
    int build{-1};
    std::string source;
};

struct TrtLibraryMatch {
    TrtVersion version;
    std::string path;
    bool already_loaded{false};
};

std::optional<TrtVersion> parse_trt_version(const std::string& text);
std::optional<TrtVersion> parse_trt_abi_tag(const std::string& text);

std::string format_trt_version(const TrtVersion& version);
std::string trt_abi_string(const TrtVersion& version);
std::string trt_abi_suffix(const TrtVersion& version);
std::string trt_backend_name_for_abi(const TrtVersion& version);

bool trt_abi_matches(const TrtVersion& lhs, const TrtVersion& rhs);
bool is_standard_trt_backend_name(const std::string& backend_name);

std::optional<TrtVersion>
detect_installed_trt_version(const std::vector<std::string>& search_dirs = {},
                             std::string* diagnostics = nullptr);
std::optional<TrtLibraryMatch>
find_trt_library_for_version(const TrtVersion& required_version,
                             const std::vector<std::string>& search_dirs = {},
                             std::string* diagnostics = nullptr);

std::vector<std::string> trt_backend_candidates(const std::string& backend_name,
                                                const std::optional<TrtVersion>& required_version,
                                                const std::optional<TrtVersion>& installed_version);

} // namespace trtmc
