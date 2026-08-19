/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sam2_bbox_postprocess.h"
#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

namespace trtmc::sam2 {

// Fixed, session-owned CUDA storage for the five-frame native SAM2 contract.
// The class is intentionally model-local: no generic runtime buffer semantics
// are changed to support this one fixed recurrent graph family.
class Sam2DeviceWorkspace final {
  public:
    explicit Sam2DeviceWorkspace(cudaStream_t stream);
    ~Sam2DeviceWorkspace();
    Sam2DeviceWorkspace(const Sam2DeviceWorkspace&) = delete;
    Sam2DeviceWorkspace& operator=(const Sam2DeviceWorkspace&) = delete;

    std::int32_t deviceOrdinal() const noexcept;

    void* historyMemoryBase() const noexcept;
    void* historyPointerBase() const noexcept;
    void* historyMemorySlot(std::size_t frame_index) const;
    void* historyPointerSlot(std::size_t frame_index) const;

    // FP32 NCHW storage bound directly to the image engine's existing
    // pixel_values input and filled by the same-stream RGB8 CUDA pipeline.
    void* preprocessedPixelValues() const noexcept;
    void enqueueRgb8Preprocess(const std::uint8_t* rgb_hwc, std::int32_t height,
                               std::int32_t width);

    void beginRun();
    void enqueueBboxDownload(const ITrtModule& image);
    Sam2BBoxRawOutputs waitForBbox();
    void enqueueTrackerPostprocess(const ITrtModule& tracker, std::size_t frame_index);
    void finishTrackerStage(const char* stage);
    void drainNoexcept() noexcept;
    const void* maskPointer(std::size_t frame_index) const;
    std::vector<std::uint8_t> materializeMask(std::size_t frame_index) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

} // namespace trtmc::sam2
