/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_video_session.h"

#include "sam2_engine_contract.h"
#include "sam2_native_video_processor.h"
#include "trtmc/models/sam2_video.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

struct TrtmcSam2VideoSession {
    explicit TrtmcSam2VideoSession(
        std::unique_ptr<trtmc::sam2::NativeVideoProcessor> native_processor)
        : processor(std::move(native_processor)) {}

    std::unique_ptr<trtmc::sam2::NativeVideoProcessor> processor;
};

namespace {

static_assert(TRTMC_SAM2_VIDEO_FRAME_COUNT_V1 == trtmc::sam2::kFrameCount);
static_assert(TRTMC_SAM2_VIDEO_FRAME_HEIGHT_V1 == trtmc::sam2::kOriginalImageHeight);
static_assert(TRTMC_SAM2_VIDEO_FRAME_WIDTH_V1 == trtmc::sam2::kOriginalImageWidth);

constexpr std::size_t kLastErrorCapacity = 1024U;
thread_local std::array<char, kLastErrorCapacity> last_error{};

void clearError() noexcept {
    last_error.front() = '\0';
}

void setError(const char* message) noexcept {
    if (message == nullptr)
        message = "unknown native exception";
    std::size_t length = 0U;
    while (length + 1U < last_error.size() && message[length] != '\0') {
        last_error[length] = message[length];
        ++length;
    }
    last_error[length] = '\0';
}

TrtmcSam2VideoSession& requireSession(TrtmcSam2VideoSession* session) {
    if (session == nullptr || session->processor == nullptr)
        throw std::invalid_argument("null Model Connect SAM2 video session");
    return *session;
}

template <typename Function>
int32_t translateErrors(Function&& function) noexcept {
    try {
        clearError();
        function();
        return TRTMC_SAM2_VIDEO_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        setError(error.what());
        return TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT;
    } catch (const std::logic_error& error) {
        setError(error.what());
        return TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE;
    } catch (const std::bad_alloc&) {
        setError("SAM2 video session allocation failed");
        return TRTMC_SAM2_VIDEO_STATUS_RUNTIME_ERROR;
    } catch (const std::exception& error) {
        setError(error.what());
        return TRTMC_SAM2_VIDEO_STATUS_RUNTIME_ERROR;
    } catch (...) {
        setError("unknown native exception");
        return TRTMC_SAM2_VIDEO_STATUS_RUNTIME_ERROR;
    }
}

} // namespace

namespace trtmc::sam2 {

TrtmcSam2VideoSession* makeVideoSessionHandle(std::unique_ptr<NativeVideoProcessor> processor) {
    if (processor == nullptr)
        throw std::invalid_argument("SAM2 native video processor is missing");
    return new TrtmcSam2VideoSession(std::move(processor));
}

namespace c_api_internal {

void clearLastError() noexcept {
    clearError();
}

void setLastError(const char* message) noexcept {
    setError(message);
}

} // namespace c_api_internal
} // namespace trtmc::sam2

extern "C" {

uint32_t trtmc_sam2_video_abi_version(void) noexcept {
    return TRTMC_SAM2_VIDEO_ABI_VERSION_1;
}

const char* trtmc_sam2_video_last_error(void) noexcept {
    return last_error.data();
}

void trtmc_sam2_video_session_destroy(TrtmcSam2VideoSession* session) noexcept {
    delete session;
}

int32_t trtmc_sam2_video_run_rgb8_v1(TrtmcSam2VideoSession* session, const uint8_t* frame0,
                                     const uint8_t* frame1, const uint8_t* frame2,
                                     const uint8_t* frame3, const uint8_t* frame4, uint32_t flags,
                                     TrtmcSam2VideoRunResultV1* result,
                                     uint64_t result_struct_size) noexcept {
    if (result == nullptr || result_struct_size < sizeof(TrtmcSam2VideoRunResultV1)) {
        setError("SAM2 version-1 run result is missing or too small");
        return TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI;
    }
    if ((flags & ~static_cast<uint32_t>(TRTMC_SAM2_VIDEO_RUN_MATERIALIZE_MASKS_HOST)) != 0U) {
        setError("SAM2 run contains unsupported version-1 flags");
        return TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI;
    }

    return translateErrors([&] {
        const trtmc::sam2::NativeRgb8Frames frames{frame0, frame1, frame2, frame3, frame4};
        const bool host = (flags & TRTMC_SAM2_VIDEO_RUN_MATERIALIZE_MASKS_HOST) != 0U;
        const auto view = requireSession(session).processor->run(frames, host);

        TrtmcSam2VideoRunResultV1 output{};
        output.struct_size = sizeof(output);
        output.abi_version = TRTMC_SAM2_VIDEO_ABI_VERSION_1;
        output.mask_memory_kind =
            host ? TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST : TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE;
        output.mask_device_ordinal = view.mask_device_ordinal;
        output.label = view.label;
        output.detector_score = view.detector_score;
        std::copy(view.prompt_box_xyxy.begin(), view.prompt_box_xyxy.end(), output.prompt_box_xyxy);
        std::copy(view.masks.begin(), view.masks.end(), output.masks);
        *result = output;
    });
}

} // extern "C"
