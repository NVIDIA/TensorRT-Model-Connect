/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2_hoi/sam2_hoi_video_session.h"

#include "runtime/models/sam2_hoi/pipeline.h"
#include "trtmc/models/sam2_hoi_video.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct TrtmcSam2HoiVideoSession {
    explicit TrtmcSam2HoiVideoSession(
        std::unique_ptr<trtmc::sam2_hoi::Sam2HoiPipeline> video_pipeline)
        : pipeline(std::move(video_pipeline)) {}

    std::unique_ptr<trtmc::sam2_hoi::Sam2HoiPipeline> pipeline;
    bool poisoned{false};
};

namespace {

static_assert(TRTMC_SAM2_HOI_VIDEO_FRAME_COUNT_V1 == 5U);

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

TrtmcSam2HoiVideoSession& requireSession(TrtmcSam2HoiVideoSession* session) {
    if (session == nullptr || session->pipeline == nullptr)
        throw std::invalid_argument("null Model Connect SAM2 HOI video session");
    return *session;
}

bool isJpegPath(const std::string& path) {
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == ".jpg" || extension == ".jpeg";
}

std::array<std::string, TRTMC_SAM2_HOI_VIDEO_FRAME_COUNT_V1> validateRunArguments(
    const std::array<const char*, TRTMC_SAM2_HOI_VIDEO_FRAME_COUNT_V1>& frame_paths,
    const char* output_json, const char* output_masks_dir) {
    std::array<std::string, TRTMC_SAM2_HOI_VIDEO_FRAME_COUNT_V1> paths;
    for (std::size_t index = 0; index < paths.size(); ++index) {
        if (frame_paths[index] == nullptr || *frame_paths[index] == '\0') {
            throw std::invalid_argument("SAM2 HOI requires five nonempty JPEG paths");
        }
        paths[index] = frame_paths[index];
        if (!isJpegPath(paths[index])) {
            throw std::invalid_argument("SAM2 HOI fixed video inputs must all be JPEG paths");
        }
    }
    if (output_json == nullptr || output_masks_dir == nullptr) {
        throw std::invalid_argument("SAM2 HOI output path pointers must not be null");
    }
    if ((*output_json == '\0') != (*output_masks_dir == '\0')) {
        throw std::invalid_argument(
            "SAM2 HOI output paths must both be empty or both be non-empty");
    }
    if (*output_json != '\0') {
        trtmc::sam2_hoi::validateVideoOutputPaths(output_json, output_masks_dir,
                                                  TRTMC_SAM2_HOI_VIDEO_FRAME_COUNT_V1);
    }
    return paths;
}

int32_t
runSession(TrtmcSam2HoiVideoSession& session,
           const std::array<std::string, TRTMC_SAM2_HOI_VIDEO_FRAME_COUNT_V1>& checked_paths,
           const char* output_json, const char* output_masks_dir) {
    if (session.poisoned)
        throw std::logic_error("SAM2 HOI video session is poisoned; recreate it");

    try {
        const std::vector<std::string> paths(checked_paths.begin(), checked_paths.end());
        auto frames = session.pipeline->load_video_frames(paths);
        if (frames.size() != TRTMC_SAM2_HOI_VIDEO_FRAME_COUNT_V1) {
            throw std::runtime_error("SAM2 HOI JPEG batch decoder returned the wrong frame count");
        }
        std::vector<trtmc::sam2_hoi::Sam2HoiVideoFrameView> views;
        views.reserve(frames.size());
        for (const auto& frame : frames)
            views.push_back(frame.view());
        return session.pipeline->track_video(views, output_json, output_masks_dir);
    } catch (...) {
        session.poisoned = true;
        throw;
    }
}

template <typename Function>
int32_t translateErrors(Function&& function) noexcept {
    try {
        clearError();
        function();
        return TRTMC_SAM2_HOI_VIDEO_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        setError(error.what());
        return TRTMC_SAM2_HOI_VIDEO_STATUS_INVALID_ARGUMENT;
    } catch (const std::logic_error& error) {
        setError(error.what());
        return TRTMC_SAM2_HOI_VIDEO_STATUS_INVALID_STATE;
    } catch (const std::bad_alloc&) {
        setError("SAM2 HOI video session allocation failed");
        return TRTMC_SAM2_HOI_VIDEO_STATUS_RUNTIME_ERROR;
    } catch (const std::exception& error) {
        setError(error.what());
        return TRTMC_SAM2_HOI_VIDEO_STATUS_RUNTIME_ERROR;
    } catch (...) {
        setError("unknown native exception");
        return TRTMC_SAM2_HOI_VIDEO_STATUS_RUNTIME_ERROR;
    }
}

} // namespace

namespace trtmc::sam2_hoi {

TrtmcSam2HoiVideoSession* makeVideoSessionHandle(std::unique_ptr<Sam2HoiPipeline> pipeline) {
    if (pipeline == nullptr)
        throw std::invalid_argument("SAM2 HOI video pipeline is missing");
    return new TrtmcSam2HoiVideoSession(std::move(pipeline));
}

namespace c_api_internal {

void clearLastError() noexcept {
    clearError();
}

void setLastError(const char* message) noexcept {
    setError(message);
}

} // namespace c_api_internal
} // namespace trtmc::sam2_hoi

extern "C" {

uint32_t trtmc_sam2_hoi_video_abi_version(void) noexcept {
    return TRTMC_SAM2_HOI_VIDEO_ABI_VERSION_1;
}

const char* trtmc_sam2_hoi_video_last_error(void) noexcept {
    return last_error.data();
}

void trtmc_sam2_hoi_video_session_destroy(TrtmcSam2HoiVideoSession* session) noexcept {
    delete session;
}

int32_t trtmc_sam2_hoi_video_run_jpeg_files_v1(
    TrtmcSam2HoiVideoSession* session, const char* frame0, const char* frame1, const char* frame2,
    const char* frame3, const char* frame4, const char* output_json, const char* output_masks_dir,
    TrtmcSam2HoiVideoRunResultV1* result, uint64_t result_struct_size) noexcept {
    if (result == nullptr || result_struct_size < sizeof(TrtmcSam2HoiVideoRunResultV1)) {
        setError("SAM2 HOI version-1 run result is missing or too small");
        return TRTMC_SAM2_HOI_VIDEO_STATUS_UNSUPPORTED_ABI;
    }

    return translateErrors([&] {
        const std::array<const char*, TRTMC_SAM2_HOI_VIDEO_FRAME_COUNT_V1> frames{
            frame0, frame1, frame2, frame3, frame4};
        const auto checked_paths = validateRunArguments(frames, output_json, output_masks_dir);
        const int32_t produced =
            runSession(requireSession(session), checked_paths, output_json, output_masks_dir);

        TrtmcSam2HoiVideoRunResultV1 output{};
        output.struct_size = sizeof(output);
        output.abi_version = TRTMC_SAM2_HOI_VIDEO_ABI_VERSION_1;
        output.produced_frame_count = produced;
        *result = output;
    });
}

} // extern "C"
