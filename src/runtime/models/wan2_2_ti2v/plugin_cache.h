/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace trtmc {

// Atomically publish one extracted CUDA plugin. Concurrent callers may target
// the same path; every successful return guarantees the final file is complete.
void publish_wan22_cuda_plugin(const std::filesystem::path& output, const std::vector<char>& bytes);

// Return an explicitly configured development override, or an empty string.
// Overrides are fail-closed unless the development gate is enabled, and are
// always rejected under strict model-plugin loading.
std::string resolve_wan22_cuda_plugin_override(const char* environment_name);

// Bind one fixed TensorRT creator set to the exact bytes that first registered
// it in this process. Reusing identical bytes is allowed; different bytes are
// rejected before their DSO can execute registration code.
void record_wan22_cuda_plugin_provenance(const std::string& creator_set,
                                         const std::vector<char>& bytes);

} // namespace trtmc
