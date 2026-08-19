/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sam2_engine_contract.h"
#include "trtmc/runtime/trt_module.h"

#include <array>
#include <cstdint>
#include <memory>

namespace trtmc::sam2 {

inline constexpr std::size_t kVideoFrameCount = static_cast<std::size_t>(kFrameCount);
using NativeRgb8Frames = std::array<const std::uint8_t*, kVideoFrameCount>;

struct NativeVideoEngineSet {
    std::unique_ptr<ITrtModule> image;
    std::unique_ptr<ITrtModule> prompt;
    std::array<std::unique_ptr<ITrtModule>, 4> recurrent;
};

struct NativeVideoRunView {
    std::int32_t label{-1};
    float detector_score{0.0F};
    std::array<float, 4> prompt_box_xyxy{};
    std::array<const void*, kVideoFrameCount> masks{};
    std::int32_t mask_device_ordinal{-1};
};

// Owns the six fixed-shape TensorRT modules and their reusable CUDA workspace.
class NativeVideoProcessor final {
  public:
    explicit NativeVideoProcessor(NativeVideoEngineSet engines);
    ~NativeVideoProcessor();

    NativeVideoProcessor(const NativeVideoProcessor&) = delete;
    NativeVideoProcessor& operator=(const NativeVideoProcessor&) = delete;

    NativeVideoRunView run(const NativeRgb8Frames& frames, bool materialize_masks_host);

  private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

} // namespace trtmc::sam2
