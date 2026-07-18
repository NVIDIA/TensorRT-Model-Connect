/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>

namespace trtmc {

class BundleSectionReader;

namespace wan2_2_ti2v {

// Validate the exact v4 source-bound bundle contract, including the embedded
// executable section, before any bundled code or TensorRT plan is loaded.
void validate_bundle_artifact_provenance(BundleSectionReader& reader,
                                         const std::string& config_json,
                                         std::size_t materialized_config_size);

} // namespace wan2_2_ti2v
} // namespace trtmc
