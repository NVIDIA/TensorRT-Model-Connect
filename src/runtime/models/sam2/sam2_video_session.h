/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/models/sam2_video.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace sam2 {
class Sam2DeviceWorkspace;
}

inline constexpr std::size_t kSam2VideoFrameCount = TRTMC_SAM2_VIDEO_FRAME_COUNT_V1;

class Sam2VideoStateError final : public std::logic_error {
  public:
    using std::logic_error::logic_error;
};

class Sam2VideoProcessorError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class Sam2VideoResultError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct Sam2VideoLimits {
    std::size_t max_frame_elements{300000000};
    bool validate_input_values{false};
};

enum class Sam2VideoPixelFormat : std::uint8_t {
    kFloat32Rgb01,
    kUint8Rgb,
};

struct Sam2VideoFrameView {
    int32_t frame_index{-1};
    int32_t height{0};
    int32_t width{0};
    const float* pixels{nullptr};
    std::size_t pixel_elements{0};
    // Appended after the original fields so existing aggregate construction
    // and the float32 API remain source-compatible.
    Sam2VideoPixelFormat pixel_format{Sam2VideoPixelFormat::kFloat32Rgb01};
    const std::uint8_t* rgb8_pixels{nullptr};
    std::size_t rgb8_bytes{0};
};

using Sam2VideoFrames = std::array<Sam2VideoFrameView, kSam2VideoFrameCount>;

class Sam2VideoMaskBuffer final {
  public:
    using HostMaterializer = std::function<std::vector<uint8_t>()>;

    static Sam2VideoMaskBuffer host(std::vector<uint8_t> mask);

    // The owner retains the CUDA allocation containing data. The producer
    // asserts that the device buffer contains uint8 values created by a binary
    // threshold operation. materialize_host is called only by an explicit host
    // result query, never by prompt or propagation validation.
    static Sam2VideoMaskBuffer cuda_device_binary(const void* data, std::size_t byte_count,
                                                  int32_t device_ordinal,
                                                  std::shared_ptr<const void> owner,
                                                  HostMaterializer materialize_host);

    TrtmcSam2VideoMaskMemoryKind memory_kind() const noexcept;
    const void* data() const noexcept;
    std::size_t byte_count() const noexcept;
    int32_t device_ordinal() const noexcept;
    bool binary_by_construction() const noexcept;
    bool has_owner() const noexcept;
    bool can_materialize_host() const noexcept;

    const std::vector<uint8_t>& materialize_host(std::size_t expected_bytes);

  private:
    friend class sam2::Sam2DeviceWorkspace;

    bool shares_device_owner(const std::shared_ptr<const void>& expected) const noexcept;

    TrtmcSam2VideoMaskMemoryKind memory_kind_{TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST};
    std::vector<uint8_t> host_mask_;
    const void* device_data_{nullptr};
    std::size_t byte_count_{0};
    int32_t device_ordinal_{-1};
    std::shared_ptr<const void> device_owner_;
    HostMaterializer materialize_host_;
    std::unique_ptr<std::vector<uint8_t>> materialized_host_mask_;
    bool binary_by_construction_{false};
};

struct Sam2VideoTrack {
    int32_t label{-1};
    float detector_score{0.0F};
    std::array<float, 4> prompt_box_xyxy{};
};

struct Sam2VideoFrameResult {
    int32_t frame_index{-1};
    int32_t height{0};
    int32_t width{0};
    Sam2VideoMaskBuffer mask{Sam2VideoMaskBuffer::host({})};
};

struct Sam2VideoPromptResult {
    Sam2VideoTrack track;
    Sam2VideoFrameResult frame_zero;
};

using Sam2VideoFrameResults = std::array<Sam2VideoFrameResult, kSam2VideoFrameCount>;

struct Sam2VideoProcessor {
    using RunBboxPrompt = std::function<Sam2VideoPromptResult(const Sam2VideoFrames&)>;
    using Propagate =
        std::function<Sam2VideoFrameResults(const Sam2VideoPromptResult&, const Sam2VideoFrames&)>;
    using Reset = std::function<void()>;

    RunBboxPrompt run_bbox_prompt;
    Propagate propagate;
    Reset reset;

    explicit operator bool() const noexcept {
        return static_cast<bool>(run_bbox_prompt) && static_cast<bool>(propagate) &&
               static_cast<bool>(reset);
    }
};

struct Sam2VideoFrameResultView {
    int32_t frame_index{-1};
    int32_t height{0};
    int32_t width{0};
    const void* mask{nullptr};
    std::size_t mask_byte_count{0};
    TrtmcSam2VideoMaskMemoryKind mask_memory_kind{TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST};
    int32_t mask_device_ordinal{-1};
};

class Sam2VideoSegmentationSession final {
  public:
    explicit Sam2VideoSegmentationSession(Sam2VideoProcessor processor,
                                          Sam2VideoLimits limits = {});

    Sam2VideoSegmentationSession(const Sam2VideoSegmentationSession&) = delete;
    Sam2VideoSegmentationSession& operator=(const Sam2VideoSegmentationSession&) = delete;

    void reset();
    void begin();
    void append_frame(const float* pixels, int32_t height, int32_t width);
    void append_frame_rgb8(const std::uint8_t* pixels, int32_t height, int32_t width);
    void run_bbox_prompt();
    std::size_t propagate();

    const Sam2VideoTrack& track() const;
    std::size_t result_count() const;
    Sam2VideoFrameResultView result(std::size_t index, bool materialize_mask_host);

  private:
    enum class State { kIdle, kCollecting, kPrompted, kComplete, kPoisoned };

    Sam2VideoProcessor processor_;
    Sam2VideoLimits limits_;
    State state_{State::kIdle};
    std::size_t appended_frames_{0};
    Sam2VideoFrames frames_{};
    std::unique_ptr<Sam2VideoPromptResult> prompt_result_;
    std::unique_ptr<Sam2VideoFrameResults> propagated_results_;
};

// Engine integration creates the opaque ABI session with a model-owned
// processor. The returned handle is released by
// trtmc_sam2_video_session_destroy().
TrtmcSam2VideoSession* make_sam2_video_session_handle(Sam2VideoProcessor processor,
                                                      Sam2VideoLimits limits = {});

namespace sam2::c_api_internal {

// Shared by the family plugin's qualified constructor and the session ABI
// implementation so both entry points report through one thread-local slot.
void clearLastError() noexcept;
void setLastError(const char* message) noexcept;

} // namespace sam2::c_api_internal

} // namespace trtmc
