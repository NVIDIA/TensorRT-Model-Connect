/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// BackendLoader: loads backend shared libraries and caches them.

#include "trtmc/runtime/trt_backend.h"

#include <string>
#include <vector>

namespace trtmc {

struct BackendLoadMetadata {
    std::string requested_name;
    std::string dso_name;
    std::string backend_name;
    std::string trt_abi;
    std::string trt_runtime_version;
};

class BackendLoader {
  public:
    // Load backend by name ("trt" or "trt_rtx").
    // Caches: second call with same name returns same IBackend*.
    // Throws std::runtime_error if the library is not found or its factory is missing.
    static IBackend* load(const std::string& backend_name);
    static IBackend* load(const std::string& backend_name,
                          const std::vector<std::string>& search_dirs);
    static IBackend* load_first_available(const std::vector<std::string>& backend_names,
                                          const std::vector<std::string>& search_dirs,
                                          std::string* loaded_backend_name = nullptr,
                                          BackendLoadMetadata* metadata = nullptr);
    static void preload_dependency(const std::string& path);
};

} // namespace trtmc
