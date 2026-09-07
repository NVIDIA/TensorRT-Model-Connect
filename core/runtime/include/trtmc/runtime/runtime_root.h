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

// Return whether runtime_root contains one complete build cohort for bundle
// that matches the core and runtime loader already active in this process.
// This function validates one explicit candidate and never searches, loads, or
// falls back to another directory.
bool runtime_root_matches_loaded_build(const BundleInfo& bundle, const std::string& runtime_root,
                                       bool require_byok = false);

} // namespace trtmc
