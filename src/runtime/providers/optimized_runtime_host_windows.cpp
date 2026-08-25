/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/providers/optimized_runtime_host.h"

#include <algorithm>
#include <stdexcept>

namespace trtmc {

bool is_optimized_runtime_bundle(const BundleInfo& bundle_info) {
    return std::any_of(
        bundle_info.sections.begin(), bundle_info.sections.end(),
        [](const BundleSectionInfo& section) { return section.name == "optimized_runtime.json"; });
}

std::unique_ptr<IPipeline> try_make_optimized_runtime_pipeline(const std::string&,
                                                               const BundleInfo& bundle_info,
                                                               const LoadOptions&) {
    if (!is_optimized_runtime_bundle(bundle_info))
        return nullptr;
    throw std::runtime_error(
        "Optimized-runtime capsules are not supported by the native Windows build");
}

} // namespace trtmc
