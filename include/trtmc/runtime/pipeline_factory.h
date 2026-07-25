/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// PipelineFactory: bundle-driven pipeline assembly.
// Optimized-runtime bundles are claimed by optimized_runtime.json and load
// their embedded implementation DSO. Native bundles read config.json,
// dispatch on runtime_strategy, and load the owning model and backend DSOs.
//
// This is the single entry point for creating pipelines from bundles.

#include "trtmc/pipeline.h"

#include <cstddef>
#include <memory>
#include <string>

namespace trtmc {

// The lease API is declared in trtmc/runtime/pipeline_pool.h and its
// synchronization implementation is linked from the runtime library.
class PipelinePool;

class PipelineFactory {
  public:
    static std::unique_ptr<IPipeline> from_bundle(const std::string& bundle_path,
                                                  const std::string& hf_python = "",
                                                  const std::string& runtime_cache_path = "",
                                                  bool cuda_graphs = false);
    static std::unique_ptr<IPipeline> from_bundle(const std::string& bundle_path,
                                                  const LoadOptions& options);
    // Native bundles only. Creates pool_size independent pipeline instances.
    // Optimized-runtime bundles are rejected because their delegated runtime
    // owns batching and scheduling.
    static std::unique_ptr<PipelinePool> from_bundle_pool(const std::string& bundle_path,
                                                          std::size_t pool_size,
                                                          const LoadOptions& options = {});
};

} // namespace trtmc
