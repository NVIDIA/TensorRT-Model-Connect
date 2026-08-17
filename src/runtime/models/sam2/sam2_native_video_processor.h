/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam2/sam2_video_session.h"
#include "trtmc/runtime/trt_module.h"

#include <array>
#include <memory>

namespace trtmc::sam2 {

// The native SAM2 v1 runtime is deliberately split into six fixed-shape
// TensorRT plans. Ownership moves into the returned Sam2VideoProcessor; no
// plan is shared with another session.
struct NativeVideoEngineSet {
    std::unique_ptr<ITrtModule> image;
    std::unique_ptr<ITrtModule> prompt;
    std::array<std::unique_ptr<ITrtModule>, 4> recurrent;
};

// Construct the host-input native processor used by focused validation. The
// engine metadata is validated immediately and before every inference phase.
// The current path uses ITrtModule::forward with direct TensorRT C++ modules
// and family-owned native preprocessing and postprocessing.
Sam2VideoProcessor makeNativeVideoProcessor(NativeVideoEngineSet engines);

// Construct the explicit device-resident native processor. All six modules
// must already use the same non-null CUDA stream and expose the exact static
// SAM2 ABI. Construction binds image FPN outputs and recurrent history storage
// once; any unsupported binding fails closed instead of falling back to the
// host oracle above.
Sam2VideoProcessor makeNativeDeviceVideoProcessor(NativeVideoEngineSet engines);

} // namespace trtmc::sam2
