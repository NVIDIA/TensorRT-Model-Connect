/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>

namespace trtmc {

class BundleReader;

// Load exactly the family and backend named in bundle_path from runtime_root.
// No environment, installed-package, current-directory, alias, or fallback
// search is performed. Loaded DSOs, backend instances, and immutable backend
// option adapters stay resident for the process lifetime so family tasks may
// safely defer module creation.
std::unique_ptr<ITask> load_task(const std::string& bundle_path, const std::string& runtime_root,
                                 std::uint64_t kv_cache_size_bytes = 0,
                                 const std::string& runtime_cache_path = {},
                                 bool cuda_graphs = false);

// Reuse an already inspected bundle so callers use the same metadata snapshot
// for reporting and family/backend selection, without parsing it a second time.
std::unique_ptr<ITask> load_task(const BundleReader& reader, const std::string& runtime_root,
                                 std::uint64_t kv_cache_size_bytes = 0,
                                 const std::string& runtime_cache_path = {},
                                 bool cuda_graphs = false);

// Register a kernel before model loading, using the existing optional TVM-FFI
// extension. Successfully loaded extension DSOs remain resident until exit.
void preload_byok_kernel(const std::string& runtime_root, const std::string& library,
                         const std::string& function, const std::string& kernel_name);

} // namespace trtmc
