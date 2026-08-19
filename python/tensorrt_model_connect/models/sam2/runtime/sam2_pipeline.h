/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sam2_native_video_processor.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/pipeline_plugin.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace trtmc::sam2 {

using NativePlanModuleFactory = std::function<std::unique_ptr<ITrtModule>(
    std::string_view section, const void* plan_data, std::size_t plan_size)>;

// Validate the small model-owned bundle contract and create its six modules.
// Tensor I/O validation stays in NativeVideoProcessor, where it is needed
// regardless of how the modules were materialized.
NativeVideoEngineSet makeNativeVideoEngineSet(const BundleFile& bundle,
                                              const NativePlanModuleFactory& module_factory);

// Thin wrapper for a locally built five-frame SAM2 runtime.
class Sam2Pipeline final : public IPipeline {
  public:
    static std::unique_ptr<Sam2Pipeline> create(const PipelineContext& context,
                                                const NativePlanModuleFactory& module_factory);

    Sam2Pipeline(const Sam2Pipeline&) = delete;
    Sam2Pipeline& operator=(const Sam2Pipeline&) = delete;

    std::unique_ptr<NativeVideoProcessor> releaseVideoProcessor();

    const char* model_id() const override;
    const char* pipeline_type() const override { return "Sam2NativeVideoPipeline"; }

  private:
    explicit Sam2Pipeline(std::unique_ptr<NativeVideoProcessor> processor);

    std::unique_ptr<NativeVideoProcessor> processor_;
};

} // namespace trtmc::sam2
