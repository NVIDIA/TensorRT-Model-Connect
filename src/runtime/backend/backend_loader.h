/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// BackendLoader: loads backend DSOs via dlopen and caches them.

#include "trtmc/runtime/trt_backend.h"

#include <string>
#include <vector>

namespace trtmc {

struct BackendLoadMetadata {
    std::string requested_name;
    std::string dso_name;
    std::string backend_name;
    std::uint32_t backend_abi_contract_version{0};
    std::uint32_t runtime_memory_backend_api_version{0};
    std::uint64_t backend_capability_flags{0};
    std::uint64_t backend_interface_fingerprint{0};
    std::uint64_t runtime_memory_layout_fingerprint{0};
    std::string trt_abi;
    std::string trt_runtime_version;
    // Independently detected by the selected backend/plugin DSO. Empty for
    // compatibility backends that do not implement native runtime-owned KV.
    std::string runtime_memory_stack_json;
};

// Shared by the backend loader and the model-plugin ABI gate. Returns an
// empty string only when every core/backend build and layout field matches.
std::string backend_dso_abi_contract_mismatch(const BackendDsoAbiContractV2& actual);

class BackendLoader {
  public:
    // Load backend by name ("trt" or "trt_rtx").
    // Caches: second call with same name returns same IBackend*.
    // Throws std::runtime_error if DSO not found or factory missing.
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
