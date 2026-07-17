/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/bundle.h"
#include "trtmc/pipeline.h"

#include <memory>
#include <string>

namespace trtmc {

// Generic, behavior-free host for model-owned optimized-runtime capsules.
//
// The presence of an optimized_runtime.json bundle section claims this path.
// A claimed bundle either initializes its exact runtime DSO successfully or
// fails closed; it never falls through to another optimized runtime or to the
// native Model Connect implementation.
std::unique_ptr<IPipeline> try_make_optimized_runtime_pipeline(const std::string& bundle_path,
                                                               const BundleInfo& bundle_info,
                                                               const LoadOptions& options);

} // namespace trtmc
