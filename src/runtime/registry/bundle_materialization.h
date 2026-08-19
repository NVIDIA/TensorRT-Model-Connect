/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "bundle/bundle_format.h"

#include <string>

namespace trtmc::detail {

// The payload view needed while constructing a pipeline. Native bundles must
// carry a non-empty config.json section. Bundles without an explicit staged
// policy read all sections; staged bundles materialize only their declared
// eager sections and keep all section metadata in BundleFile::info for later
// path-based section reads.
struct PipelineBundleMaterialization {
    BundleFile bundle;
    std::string config_text;
};

PipelineBundleMaterialization materialize_pipeline_bundle(const std::string& bundle_path);

} // namespace trtmc::detail
