/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam2/sam2_native_bundle_loader.h"
#include "trtmc/pipeline.h"

#include <memory>
#include <string>

namespace trtmc::sam2 {

// Thin production admission wrapper for the five-frame SAM2 runtime. Engine
// loading is intentionally restricted to the pinned qualification path; this
// class has no diagnostic loader or sidecar discovery fallback.
class Sam2Pipeline final : public IPipeline {
  public:
    static std::unique_ptr<Sam2Pipeline>
    createProductionQualified(const std::string& bundle_path,
                              const std::string& qualification_record_path,
                              const NativeBundleRuntimeTarget& runtime_target,
                              const NativePlanModuleFactory& module_factory, std::string model_id);

    Sam2Pipeline(const Sam2Pipeline&) = delete;
    Sam2Pipeline& operator=(const Sam2Pipeline&) = delete;

    // The C ABI consumes the only processor owned by this pipeline. A second
    // extraction fails closed rather than sharing mutable tracker state.
    Sam2VideoProcessor releaseVideoProcessor();

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "Sam2NativeVideoPipeline"; }

  private:
    Sam2Pipeline(Sam2VideoProcessor processor, std::string model_id);

    Sam2VideoProcessor processor_;
    std::string model_id_;
};

} // namespace trtmc::sam2
