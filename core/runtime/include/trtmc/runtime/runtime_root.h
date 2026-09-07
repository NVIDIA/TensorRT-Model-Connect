/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/bundle.h"

#include <string>

namespace trtmc {

// Return the directory containing the runtime loader used by this process.
// Applications may use this as a discovery candidate; load_task remains the
// only operation that loads a family and backend.
std::string loaded_runtime_root();

// Return whether runtime_root contains the root-local backend and family DSOs
// named by bundle, plus BYOK when requested. This structural check never
// searches, loads, or falls back. Exact product-build and plugin identities are
// validated when the selected DSOs are loaded.
bool runtime_root_contains_bundle(const BundleInfo& bundle, const std::string& runtime_root,
                                  bool require_byok = false);

} // namespace trtmc
