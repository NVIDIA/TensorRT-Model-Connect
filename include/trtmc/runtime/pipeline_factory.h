/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// PipelineFactory: config-driven pipeline assembly.
// Reads config.json from a .trtfb bundle, dispatches on runtime_strategy,
// loads TRT engines, creates tokenizers, and assembles the appropriate pipeline.
//
// This is the single entry point for creating pipelines from bundles.

#include "trtmc/pipeline.h"

#include <memory>
#include <string>

namespace trtmc {

class PipelineFactory {
  public:
    static std::unique_ptr<IPipeline> from_bundle(const std::string& bundle_path,
                                                  const std::string& hf_python = "",
                                                  const std::string& runtime_cache_path = "",
                                                  bool cuda_graphs = false);
    static std::unique_ptr<IPipeline> from_bundle(const std::string& bundle_path,
                                                  const LoadOptions& options);
};

} // namespace trtmc
