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

std::vector<std::uint8_t> convertSam2FloatFrameToRgb8(const VideoFrameView& frame);

// Non-owning view of the fixed five device masks produced by one SAM2 run.
// Each pointer addresses kOriginalImageHeight * kOriginalImageWidth uint8 values
// on mask_device_ordinal. The view remains valid until the next call on the same
// session or until that session is destroyed. Calls on one session must be serialized.
struct Sam2DeviceMaskResultView {
    std::array<const void*, kVideoFrameCount> masks{};
    std::int32_t mask_device_ordinal{-1};
    std::int32_t label{-1};
    float detector_score{0.0F};
    std::array<float, 4> prompt_box_xyxy{};
};

class ISam2DeviceMaskSession {
  public:
    virtual ~ISam2DeviceMaskSession() = default;
    virtual Sam2DeviceMaskResultView segment_device(const VideoSegmentationRequest& request) = 0;
};

class Sam2Pipeline final : public IVideoSegmentation {
  public:
    explicit Sam2Pipeline(std::unique_ptr<NativeVideoProcessor> processor);

    std::unique_ptr<IVideoSegmentationSession> create_video_segmentation_session() override;

  private:
    std::unique_ptr<NativeVideoProcessor> processor_;
};

} // namespace trtmc::sam2
