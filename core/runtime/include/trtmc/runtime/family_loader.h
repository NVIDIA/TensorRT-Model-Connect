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
// search is performed. Loaded DSOs and backend instances stay resident for the
// process lifetime, so the returned task needs no task-specific ownership proxy.
std::unique_ptr<ITask> load_task(const std::string& bundle_path, const std::string& runtime_root,
                                 std::uint64_t kv_cache_size_bytes = 0,
                                 const std::string& runtime_cache_path = {},
                                 bool cuda_graphs = false);

} // namespace trtmc
