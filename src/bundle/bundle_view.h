/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Lightweight helpers for looking up sections in a BundleFile by name.
// Used by pipeline plugins to extract their own sections without depending
// on the monolithic BundleSections struct.

#include "bundle/bundle_format.h"

#include <string>
#include <vector>

namespace trtmc {

// Find a single section by exact name. Returns nullptr if not found.
const std::vector<char>* find_section(const BundleFile& bundle, const std::string& name);

// Find all sections whose names start with the given prefix.
// Returns pointers sorted by section name (for deterministic ordering).
std::vector<const std::vector<char>*> find_sections_by_prefix(
    const BundleFile& bundle, const std::string& prefix);

} // namespace trtmc
