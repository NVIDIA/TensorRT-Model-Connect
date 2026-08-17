/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_video_session.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace trtmc {

namespace {

std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char* message) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
        throw std::overflow_error(message);
    return lhs * rhs;
}

std::size_t checked_frame_elements(int32_t height, int32_t width, const Sam2VideoLimits& limits) {
    if (height <= 0 || width <= 0)
        throw std::invalid_argument("SAM2 video frame dimensions must be positive");
    const auto area =
        checked_product(static_cast<std::size_t>(height), static_cast<std::size_t>(width),
                        "SAM2 video frame dimensions overflow");
    const auto elements = checked_product(area, 3U, "SAM2 video RGB buffer size overflows");
    if (elements > limits.max_frame_elements)
        throw std::invalid_argument("SAM2 video frame exceeds the configured element limit");
    return elements;
}

std::size_t checked_mask_bytes(int32_t height, int32_t width) {
    if (height <= 0 || width <= 0)
        throw Sam2VideoResultError("SAM2 result dimensions must be positive");
    return checked_product(static_cast<std::size_t>(height), static_cast<std::size_t>(width),
                           "SAM2 result mask area overflows");
}

bool is_binary(uint8_t value) {
    return value == uint8_t{0} || value == uint8_t{1};
}

void validate_binary_host_mask(const std::vector<uint8_t>& mask, std::size_t expected_bytes) {
    if (mask.size() != expected_bytes)
        throw Sam2VideoResultError("SAM2 result mask buffer has the wrong size");
    if (!std::all_of(mask.begin(), mask.end(), is_binary))
        throw Sam2VideoResultError("SAM2 result mask must contain only binary uint8 values");
}

void validate_track(const Sam2VideoTrack& track) {
    if (track.label < 0)
        throw Sam2VideoResultError("SAM2 selected-track label must be non-negative");
    if (!std::isfinite(track.detector_score) || track.detector_score < 0.0F ||
        track.detector_score > 1.0F) {
        throw Sam2VideoResultError(
            "SAM2 selected-track detector score must be a finite probability");
    }

    const auto& box = track.prompt_box_xyxy;
    if (!std::all_of(box.begin(), box.end(),
                     [](float coordinate) { return std::isfinite(coordinate); })) {
        throw Sam2VideoResultError("SAM2 selected-track prompt box must be finite");
    }
    if (box[0] > box[2] || box[1] > box[3]) {
        throw Sam2VideoResultError(
            "SAM2 selected-track prompt box must use ordered absolute xyxy coordinates");
    }
}

void validate_frame_result(const Sam2VideoFrameView& frame, Sam2VideoFrameResult& result) {
    if (result.frame_index != frame.frame_index || result.height != frame.height ||
        result.width != frame.width) {
        throw Sam2VideoResultError("SAM2 result geometry does not match its input frame");
    }

    const auto mask_bytes = checked_mask_bytes(frame.height, frame.width);
    if (result.mask.byte_count() != mask_bytes)
        throw Sam2VideoResultError("SAM2 result mask byte count does not match its geometry");
    if (result.mask.data() == nullptr)
        throw Sam2VideoResultError("SAM2 result mask buffer must not be null");

    if (result.mask.memory_kind() == TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST) {
        (void)result.mask.materialize_host(mask_bytes);
        if (result.mask.device_ordinal() != -1)
            throw Sam2VideoResultError("SAM2 host masks must not name a CUDA device");
        return;
    }
    if (result.mask.memory_kind() != TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE)
        throw Sam2VideoResultError("SAM2 result uses an unknown mask memory kind");
    if (result.mask.device_ordinal() < 0 || !result.mask.has_owner() ||
        !result.mask.can_materialize_host() || !result.mask.binary_by_construction()) {
        throw Sam2VideoResultError("SAM2 device masks require an owner, device ordinal, binary "
                                   "producer, and host materializer");
    }
}

std::string processor_failure(const char* operation, const std::exception& error) {
    return std::string("SAM2 ") + operation + " processor failed: " + error.what();
}

} // namespace

Sam2VideoMaskBuffer Sam2VideoMaskBuffer::host(std::vector<uint8_t> mask) {
    Sam2VideoMaskBuffer buffer;
    buffer.memory_kind_ = TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST;
    buffer.byte_count_ = mask.size();
    buffer.host_mask_ = std::move(mask);
    buffer.binary_by_construction_ = false;
    return buffer;
}

Sam2VideoMaskBuffer Sam2VideoMaskBuffer::cuda_device_binary(const void* data,
                                                            std::size_t byte_count,
                                                            int32_t device_ordinal,
                                                            std::shared_ptr<const void> owner,
                                                            HostMaterializer materialize_host) {
    Sam2VideoMaskBuffer buffer;
    buffer.memory_kind_ = TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE;
    buffer.device_data_ = data;
    buffer.byte_count_ = byte_count;
    buffer.device_ordinal_ = device_ordinal;
    buffer.device_owner_ = std::move(owner);
    buffer.materialize_host_ = std::move(materialize_host);
    buffer.binary_by_construction_ = true;
    return buffer;
}

TrtmcSam2VideoMaskMemoryKind Sam2VideoMaskBuffer::memory_kind() const noexcept {
    return memory_kind_;
}

const void* Sam2VideoMaskBuffer::data() const noexcept {
    if (memory_kind_ == TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST)
        return host_mask_.empty() ? nullptr : host_mask_.data();
    return device_data_;
}

std::size_t Sam2VideoMaskBuffer::byte_count() const noexcept {
    return byte_count_;
}

int32_t Sam2VideoMaskBuffer::device_ordinal() const noexcept {
    return device_ordinal_;
}

bool Sam2VideoMaskBuffer::binary_by_construction() const noexcept {
    return binary_by_construction_;
}

bool Sam2VideoMaskBuffer::has_owner() const noexcept {
    return static_cast<bool>(device_owner_);
}

bool Sam2VideoMaskBuffer::can_materialize_host() const noexcept {
    return static_cast<bool>(materialize_host_);
}

bool Sam2VideoMaskBuffer::shares_device_owner(
    const std::shared_ptr<const void>& expected) const noexcept {
    if (device_owner_ == nullptr || expected == nullptr)
        return false;
    return !device_owner_.owner_before(expected) && !expected.owner_before(device_owner_);
}

const std::vector<uint8_t>& Sam2VideoMaskBuffer::materialize_host(std::size_t expected_bytes) {
    if (memory_kind_ == TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST) {
        validate_binary_host_mask(host_mask_, expected_bytes);
        return host_mask_;
    }
    if (materialized_host_mask_ != nullptr) {
        validate_binary_host_mask(*materialized_host_mask_, expected_bytes);
        return *materialized_host_mask_;
    }
    if (!materialize_host_)
        throw Sam2VideoResultError("SAM2 device mask does not have a host materializer");

    std::vector<uint8_t> materialized;
    try {
        materialized = materialize_host_();
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& error) {
        throw Sam2VideoProcessorError(processor_failure("mask materialization", error));
    } catch (...) {
        throw Sam2VideoProcessorError(
            "SAM2 mask materialization processor failed with an unknown exception");
    }
    validate_binary_host_mask(materialized, expected_bytes);
    materialized_host_mask_ = std::make_unique<std::vector<uint8_t>>(std::move(materialized));
    return *materialized_host_mask_;
}

Sam2VideoSegmentationSession::Sam2VideoSegmentationSession(Sam2VideoProcessor processor,
                                                           Sam2VideoLimits limits)
    : processor_(std::move(processor)), limits_(limits) {
    if (!processor_)
        throw std::invalid_argument("SAM2 video processor callbacks are required");
    if (limits_.max_frame_elements == 0)
        throw std::invalid_argument("SAM2 video frame-element limit must be positive");
}

void Sam2VideoSegmentationSession::reset() {
    // Invalidate all borrowed wrapper-owned views before invoking model-owned
    // cleanup. If that cleanup fails, the session stays poisoned and no stale
    // result remains queryable.
    propagated_results_.reset();
    prompt_result_.reset();
    frames_ = {};
    appended_frames_ = 0;
    state_ = State::kPoisoned;
    try {
        processor_.reset();
    } catch (const std::bad_alloc&) {
        state_ = State::kPoisoned;
        throw;
    } catch (const std::exception& error) {
        state_ = State::kPoisoned;
        throw Sam2VideoProcessorError(processor_failure("reset", error));
    } catch (...) {
        state_ = State::kPoisoned;
        throw Sam2VideoProcessorError("SAM2 reset processor failed with an unknown exception");
    }
    state_ = State::kIdle;
}

void Sam2VideoSegmentationSession::begin() {
    if (state_ != State::kIdle)
        throw Sam2VideoStateError("SAM2 video begin requires an idle session");
    state_ = State::kCollecting;
}

void Sam2VideoSegmentationSession::append_frame(const float* pixels, int32_t height,
                                                int32_t width) {
    if (state_ != State::kCollecting)
        throw Sam2VideoStateError("SAM2 frames can only be appended after begin");
    if (appended_frames_ >= kSam2VideoFrameCount)
        throw Sam2VideoStateError("SAM2 session already has exactly five frames");
    if (pixels == nullptr)
        throw std::invalid_argument("SAM2 video frame pixels must not be null");
    const auto elements = checked_frame_elements(height, width, limits_);
    if (appended_frames_ != 0 &&
        (height != frames_.front().height || width != frames_.front().width)) {
        throw std::invalid_argument("SAM2 video frames must have identical geometry");
    }
    if (appended_frames_ != 0 &&
        frames_.front().pixel_format != Sam2VideoPixelFormat::kFloat32Rgb01) {
        throw std::invalid_argument("SAM2 video runs cannot mix float32 and RGB8 frames");
    }
    if (limits_.validate_input_values) {
        for (std::size_t index = 0; index < elements; ++index) {
            const auto value = pixels[index];
            if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
                throw std::invalid_argument(
                    "SAM2 decoded RGB pixels must be finite values in the [0, 1] range");
            }
        }
    }
    frames_[appended_frames_] = {static_cast<int32_t>(appended_frames_), height, width, pixels,
                                 elements};
    ++appended_frames_;
}

void Sam2VideoSegmentationSession::append_frame_rgb8(const std::uint8_t* pixels, int32_t height,
                                                     int32_t width) {
    if (state_ != State::kCollecting)
        throw Sam2VideoStateError("SAM2 frames can only be appended after begin");
    if (appended_frames_ >= kSam2VideoFrameCount)
        throw Sam2VideoStateError("SAM2 session already has exactly five frames");
    if (pixels == nullptr)
        throw std::invalid_argument("SAM2 video RGB8 frame pixels must not be null");
    const auto elements = checked_frame_elements(height, width, limits_);
    if (appended_frames_ != 0 &&
        (height != frames_.front().height || width != frames_.front().width)) {
        throw std::invalid_argument("SAM2 video frames must have identical geometry");
    }
    if (appended_frames_ != 0 && frames_.front().pixel_format != Sam2VideoPixelFormat::kUint8Rgb) {
        throw std::invalid_argument("SAM2 video runs cannot mix float32 and RGB8 frames");
    }
    frames_[appended_frames_] = {
        static_cast<int32_t>(appended_frames_), height, width,   nullptr, 0U,
        Sam2VideoPixelFormat::kUint8Rgb,        pixels, elements};
    ++appended_frames_;
}

void Sam2VideoSegmentationSession::run_bbox_prompt() {
    if (state_ != State::kCollecting || appended_frames_ != kSam2VideoFrameCount)
        throw Sam2VideoStateError("SAM2 bbox prompt requires exactly five appended frames");

    Sam2VideoPromptResult prompt;
    try {
        prompt = processor_.run_bbox_prompt(frames_);
    } catch (const std::bad_alloc&) {
        state_ = State::kPoisoned;
        throw;
    } catch (const std::exception& error) {
        state_ = State::kPoisoned;
        throw Sam2VideoProcessorError(processor_failure("bbox prompt", error));
    } catch (...) {
        state_ = State::kPoisoned;
        throw Sam2VideoProcessorError(
            "SAM2 bbox prompt processor failed with an unknown exception");
    }

    try {
        validate_track(prompt.track);
        validate_frame_result(frames_.front(), prompt.frame_zero);
        prompt_result_ = std::make_unique<Sam2VideoPromptResult>(std::move(prompt));
    } catch (...) {
        // The processor consumed the one-shot prompt operation. Do not expose a
        // retryable collecting state if validating or retaining its output
        // fails, including allocation failure.
        state_ = State::kPoisoned;
        throw;
    }
    state_ = State::kPrompted;
}

std::size_t Sam2VideoSegmentationSession::propagate() {
    if (state_ != State::kPrompted || prompt_result_ == nullptr)
        throw Sam2VideoStateError("SAM2 propagation requires one unconsumed bbox prompt");

    Sam2VideoFrameResults results;
    try {
        results = processor_.propagate(*prompt_result_, frames_);
    } catch (const std::bad_alloc&) {
        state_ = State::kPoisoned;
        throw;
    } catch (const std::exception& error) {
        state_ = State::kPoisoned;
        throw Sam2VideoProcessorError(processor_failure("propagation", error));
    } catch (...) {
        state_ = State::kPoisoned;
        throw Sam2VideoProcessorError(
            "SAM2 propagation processor failed with an unknown exception");
    }

    try {
        for (std::size_t index = 0; index < kSam2VideoFrameCount; ++index)
            validate_frame_result(frames_[index], results[index]);
        propagated_results_ = std::make_unique<Sam2VideoFrameResults>(std::move(results));
    } catch (...) {
        // Propagation is also one-shot. Any invalid result or inability to
        // retain it requires reset before another attempt.
        state_ = State::kPoisoned;
        throw;
    }

    state_ = State::kComplete;
    return kSam2VideoFrameCount;
}

const Sam2VideoTrack& Sam2VideoSegmentationSession::track() const {
    if ((state_ == State::kPrompted || state_ == State::kComplete) && prompt_result_ != nullptr)
        return prompt_result_->track;
    if (state_ == State::kPoisoned)
        throw Sam2VideoStateError("SAM2 video session is poisoned; reset is required");
    throw Sam2VideoStateError("SAM2 selected track is unavailable before the bbox prompt");
}

std::size_t Sam2VideoSegmentationSession::result_count() const {
    if (state_ == State::kPrompted)
        return 1;
    if (state_ == State::kComplete)
        return kSam2VideoFrameCount;
    if (state_ == State::kPoisoned)
        throw Sam2VideoStateError("SAM2 video session is poisoned; reset is required");
    throw Sam2VideoStateError("SAM2 results are unavailable before the bbox prompt");
}

Sam2VideoFrameResultView Sam2VideoSegmentationSession::result(std::size_t index,
                                                              bool materialize_mask_host) {
    Sam2VideoFrameResult* selected = nullptr;
    if (state_ == State::kPrompted) {
        if (index != 0 || prompt_result_ == nullptr)
            throw std::out_of_range("SAM2 result index is out of range");
        selected = &prompt_result_->frame_zero;
    } else if (state_ == State::kComplete) {
        if (index >= kSam2VideoFrameCount || propagated_results_ == nullptr)
            throw std::out_of_range("SAM2 result index is out of range");
        selected = &(*propagated_results_)[index];
    } else if (state_ == State::kPoisoned) {
        throw Sam2VideoStateError("SAM2 video session is poisoned; reset is required");
    } else {
        throw Sam2VideoStateError("SAM2 results are unavailable before the bbox prompt");
    }

    const auto expected_mask_bytes = checked_mask_bytes(selected->height, selected->width);
    const void* mask_data = selected->mask.data();
    auto mask_kind = selected->mask.memory_kind();
    int32_t device_ordinal = selected->mask.device_ordinal();
    if (materialize_mask_host) {
        try {
            const auto& host_mask = selected->mask.materialize_host(expected_mask_bytes);
            mask_data = host_mask.data();
            mask_kind = TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST;
            device_ordinal = -1;
        } catch (...) {
            state_ = State::kPoisoned;
            throw;
        }
    }

    return {selected->frame_index, selected->height, selected->width, mask_data,
            expected_mask_bytes,   mask_kind,        device_ordinal};
}

} // namespace trtmc

struct TrtmcSam2VideoSession {
    explicit TrtmcSam2VideoSession(trtmc::Sam2VideoProcessor processor,
                                   trtmc::Sam2VideoLimits limits)
        : implementation(std::move(processor), limits) {}

    trtmc::Sam2VideoSegmentationSession implementation;
};

namespace trtmc {

TrtmcSam2VideoSession* make_sam2_video_session_handle(Sam2VideoProcessor processor,
                                                      Sam2VideoLimits limits) {
    return new TrtmcSam2VideoSession(std::move(processor), limits);
}

} // namespace trtmc

namespace {

constexpr std::size_t kSam2VideoLastErrorCapacity = 1024;
thread_local std::array<char, kSam2VideoLastErrorCapacity> sam2_video_last_error{};

void clear_last_error() noexcept {
    sam2_video_last_error.front() = '\0';
}

void set_last_error(const char* message) noexcept {
    if (message == nullptr)
        message = "unknown native exception";

    std::size_t index = 0;
    while (index + 1U < sam2_video_last_error.size() && message[index] != '\0') {
        sam2_video_last_error[index] = message[index];
        ++index;
    }
    sam2_video_last_error[index] = '\0';
}

TrtmcSam2VideoSession& require_session(TrtmcSam2VideoSession* session) {
    if (session == nullptr)
        throw std::invalid_argument("null Model Connect SAM2 video session");
    return *session;
}

const TrtmcSam2VideoSession& require_session(const TrtmcSam2VideoSession* session) {
    if (session == nullptr)
        throw std::invalid_argument("null Model Connect SAM2 video session");
    return *session;
}

template <typename Function>
int32_t translate_errors(Function&& function) noexcept {
    try {
        clear_last_error();
        function();
        return TRTMC_SAM2_VIDEO_STATUS_OK;
    } catch (const trtmc::Sam2VideoStateError& error) {
        set_last_error(error.what());
        return TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE;
    } catch (const trtmc::Sam2VideoProcessorError& error) {
        set_last_error(error.what());
        return TRTMC_SAM2_VIDEO_STATUS_PROCESSOR_ERROR;
    } catch (const trtmc::Sam2VideoResultError& error) {
        set_last_error(error.what());
        return TRTMC_SAM2_VIDEO_STATUS_INVALID_RESULT;
    } catch (const std::out_of_range& error) {
        set_last_error(error.what());
        return TRTMC_SAM2_VIDEO_STATUS_OUT_OF_RANGE;
    } catch (const std::overflow_error& error) {
        set_last_error(error.what());
        return TRTMC_SAM2_VIDEO_STATUS_OVERFLOW;
    } catch (const std::invalid_argument& error) {
        set_last_error(error.what());
        return TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT;
    } catch (const std::bad_alloc&) {
        set_last_error("SAM2 video session allocation failed");
        return TRTMC_SAM2_VIDEO_STATUS_INTERNAL_ERROR;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return TRTMC_SAM2_VIDEO_STATUS_INTERNAL_ERROR;
    } catch (...) {
        set_last_error("unknown native exception");
        return TRTMC_SAM2_VIDEO_STATUS_INTERNAL_ERROR;
    }
}

} // namespace

namespace trtmc::sam2::c_api_internal {

void clearLastError() noexcept {
    clear_last_error();
}

void setLastError(const char* message) noexcept {
    set_last_error(message);
}

} // namespace trtmc::sam2::c_api_internal

extern "C" {

uint32_t trtmc_sam2_video_abi_version(void) noexcept {
    return TRTMC_SAM2_VIDEO_ABI_VERSION_1;
}

const char* trtmc_sam2_video_last_error(void) noexcept {
    return sam2_video_last_error.data();
}

TrtmcSam2VideoSession* trtmc_sam2_video_create_from_bundle_v1(const char* bundle_path,
                                                              const char* plugin_dir,
                                                              const char* backend_dir) noexcept {
    if (bundle_path == nullptr || plugin_dir == nullptr || backend_dir == nullptr ||
        *bundle_path == '\0' || *plugin_dir == '\0' || *backend_dir == '\0') {
        set_last_error("SAM2 bundle, plugin, and backend paths are required");
        return nullptr;
    }
    set_last_error("SAM2 legacy bundle constructor is unavailable; use the qualified constructor "
                   "with an explicit qualification record");
    return nullptr;
}

void trtmc_sam2_video_session_destroy(TrtmcSam2VideoSession* session) noexcept {
    delete session;
}

int32_t trtmc_sam2_video_reset_v1(TrtmcSam2VideoSession* session) noexcept {
    return translate_errors([&] { require_session(session).implementation.reset(); });
}

int32_t trtmc_sam2_video_begin_v1(TrtmcSam2VideoSession* session) noexcept {
    return translate_errors([&] { require_session(session).implementation.begin(); });
}

int32_t trtmc_sam2_video_append_frame_v1(TrtmcSam2VideoSession* session, const float* pixels,
                                         int32_t height, int32_t width) noexcept {
    return translate_errors(
        [&] { require_session(session).implementation.append_frame(pixels, height, width); });
}

int32_t trtmc_sam2_video_append_frame_rgb8_v1(TrtmcSam2VideoSession* session, const uint8_t* pixels,
                                              int32_t height, int32_t width) noexcept {
    return translate_errors(
        [&] { require_session(session).implementation.append_frame_rgb8(pixels, height, width); });
}

int32_t trtmc_sam2_video_run_bbox_prompt_v1(TrtmcSam2VideoSession* session) noexcept {
    return translate_errors([&] { require_session(session).implementation.run_bbox_prompt(); });
}

int32_t trtmc_sam2_video_propagate_v1(TrtmcSam2VideoSession* session,
                                      uint64_t* frame_count) noexcept {
    return translate_errors([&] {
        if (frame_count == nullptr)
            throw std::invalid_argument("SAM2 propagation frame-count output must not be null");
        *frame_count = static_cast<uint64_t>(require_session(session).implementation.propagate());
    });
}

int32_t trtmc_sam2_video_get_track_v1(const TrtmcSam2VideoSession* session,
                                      TrtmcSam2VideoTrackV1* track,
                                      uint64_t track_struct_size) noexcept {
    if (track == nullptr || track_struct_size < sizeof(TrtmcSam2VideoTrackV1)) {
        set_last_error("SAM2 version-1 track structure is missing or too small");
        return TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI;
    }
    return translate_errors([&] {
        const auto& selected = require_session(session).implementation.track();
        TrtmcSam2VideoTrackV1 output{};
        output.struct_size = sizeof(output);
        output.abi_version = TRTMC_SAM2_VIDEO_ABI_VERSION_1;
        output.flags = TRTMC_SAM2_VIDEO_TRACK_PROMPT_BOX_ABSOLUTE_XYXY;
        output.label = selected.label;
        output.detector_score = selected.detector_score;
        std::copy(selected.prompt_box_xyxy.begin(), selected.prompt_box_xyxy.end(),
                  output.prompt_box_xyxy);
        *track = output;
    });
}

int32_t trtmc_sam2_video_result_count_v1(const TrtmcSam2VideoSession* session,
                                         uint64_t* frame_count) noexcept {
    return translate_errors([&] {
        if (frame_count == nullptr)
            throw std::invalid_argument("SAM2 result-count output must not be null");
        *frame_count =
            static_cast<uint64_t>(require_session(session).implementation.result_count());
    });
}

int32_t trtmc_sam2_video_get_frame_result_v1(TrtmcSam2VideoSession* session, uint64_t result_index,
                                             uint32_t get_flags,
                                             TrtmcSam2VideoFrameResultV1* result,
                                             uint64_t result_struct_size) noexcept {
    if (result == nullptr || result_struct_size < sizeof(TrtmcSam2VideoFrameResultV1)) {
        set_last_error("SAM2 version-1 frame-result structure is missing or too small");
        return TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI;
    }
    if ((get_flags & ~static_cast<uint32_t>(TRTMC_SAM2_VIDEO_GET_MATERIALIZE_MASK_HOST)) != 0U) {
        set_last_error("SAM2 result query contains unsupported version-1 flags");
        return TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI;
    }
    return translate_errors([&] {
        if (result_index > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()))
            throw std::out_of_range("SAM2 result index exceeds the host size range");
        const bool materialize = (get_flags & TRTMC_SAM2_VIDEO_GET_MATERIALIZE_MASK_HOST) != 0U;
        const auto view = require_session(session).implementation.result(
            static_cast<std::size_t>(result_index), materialize);

        TrtmcSam2VideoFrameResultV1 output{};
        output.struct_size = sizeof(output);
        output.abi_version = TRTMC_SAM2_VIDEO_ABI_VERSION_1;
        output.flags = TRTMC_SAM2_VIDEO_FRAME_MASK_BINARY;
        output.frame_index = view.frame_index;
        output.height = view.height;
        output.width = view.width;
        output.mask_device_ordinal = view.mask_device_ordinal;
        output.mask = view.mask;
        output.mask_byte_count = static_cast<uint64_t>(view.mask_byte_count);
        output.mask_row_stride_bytes = static_cast<uint64_t>(view.width);
        output.mask_memory_kind = static_cast<uint32_t>(view.mask_memory_kind);
        *result = output;
    });
}

} // extern "C"
