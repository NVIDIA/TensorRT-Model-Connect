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

// Load exactly the family and backend named in bundle_path from runtime_root.
// No environment, installed-package, current-directory, alias, or fallback
// search is performed. Loaded DSOs, backend instances, and immutable backend
// option adapters stay resident for the process lifetime so family tasks may
// safely defer module creation.
std::unique_ptr<ITask> load_task(const std::string& bundle_path, const std::string& runtime_root,
                                 std::uint64_t kv_cache_size_bytes = 0,
                                 const std::string& runtime_cache_path = {},
                                 bool cuda_graphs = false);

// Load the exact-build-checked TVM-FFI runtime extension from runtime_root, then
// publish one BYOK kernel. The extension remains resident for process lifetime.
void load_byok_kernel_from_runtime(const std::string& runtime_root, const std::string& library,
                                   const std::string& function, const std::string& kernel_name);

} // namespace trtmc
