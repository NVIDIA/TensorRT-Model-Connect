/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_backend.h"

#include <vector>

namespace trtmc::hstu {

// Validate the artifact contract against a serving target. This CPU-only check
// does not select an implementation or change the graph/cache layout.
void validate_native_attention_manifest(const std::vector<char>& manifest,
                                        bool history_cache_enabled, int major, int minor);

// Validate the model-owned specialization and retain a private materialized DSO
// through the runtime registry's lifetime. Loading remains the backend's job.
ModulePluginLibrary native_attention_library(const std::vector<char>& manifest,
                                             const std::vector<char>& library,
                                             bool history_cache_enabled = true);

} // namespace trtmc::hstu
