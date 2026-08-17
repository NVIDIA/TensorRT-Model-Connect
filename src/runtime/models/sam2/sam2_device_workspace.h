/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam2/sam2_bbox_postprocess.h"
#include "runtime/models/sam2/sam2_video_session.h"
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
class Sam2DeviceWorkspace final : public std::enable_shared_from_this<Sam2DeviceWorkspace> {
  public:
    static std::shared_ptr<Sam2DeviceWorkspace> create(cudaStream_t stream);

    ~Sam2DeviceWorkspace();
    Sam2DeviceWorkspace(const Sam2DeviceWorkspace&) = delete;
    Sam2DeviceWorkspace& operator=(const Sam2DeviceWorkspace&) = delete;

    cudaStream_t stream() const noexcept;
    std::int32_t deviceOrdinal() const noexcept;

    void* historyMemoryBase() const noexcept;
    void* historyPointerBase() const noexcept;
    void* historyMemorySlot(std::size_t frame_index) const;
    void* historyPointerSlot(std::size_t frame_index) const;

    // FP32 NCHW storage bound directly to the image engine's existing
    // pixel_values input. Float callers still upload through ITrtModule;
    // RGB8 callers fill it with the same-stream CUDA preprocess pipeline.
    void* preprocessedPixelValues() const noexcept;
    void enqueueRgb8Preprocess(const std::uint8_t* rgb_hwc, std::int32_t height,
                               std::int32_t width);

    void beginRun();
    void enqueueBboxDownload(const ITrtModule& image);
    Sam2BBoxRawOutputs waitForBbox();
    void enqueueTrackerPostprocess(const ITrtModule& tracker, std::size_t frame_index);
    void finishTrackerStage(const char* stage);
    void drain();
    void drainNoexcept() noexcept;
    void invalidateRun() noexcept;

    bool isDeviceSpan(const void* pointer, std::size_t bytes) const noexcept;

    Sam2VideoMaskBuffer makeMaskBuffer(std::size_t frame_index);
    bool ownsMask(const Sam2VideoMaskBuffer& mask, std::size_t frame_index) const noexcept;

  private:
    struct Impl;

    explicit Sam2DeviceWorkspace(cudaStream_t stream);
    std::vector<std::uint8_t> materializeMask(std::size_t frame_index,
                                              const std::shared_ptr<const void>& provenance) const;
    const void* maskPointer(std::size_t frame_index) const noexcept;

    std::unique_ptr<Impl> implementation_;
};

} // namespace trtmc::sam2
