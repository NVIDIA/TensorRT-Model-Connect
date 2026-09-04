/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/sam2/runtime/sam2_native_video_processor.h"
#include "trtmc/bundle.h"
#include "trtmc/task.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string_view>

namespace trtmc::sam2 {

using NativePlanModuleFactory = std::function<std::unique_ptr<ITrtModule>(
    std::string_view section, const void* plan_data, std::size_t plan_size)>;

NativeVideoEngineSet makeNativeVideoEngineSet(const BundleReader& bundle,
                                              const NativePlanModuleFactory& module_factory);

class Sam2Pipeline final : public IVideoSegmentation {
  public:
    explicit Sam2Pipeline(std::unique_ptr<NativeVideoProcessor> processor);

    std::unique_ptr<IVideoSegmentationSession> create_video_segmentation_session() override;

  private:
    std::unique_ptr<NativeVideoProcessor> processor_;
};

} // namespace trtmc::sam2
