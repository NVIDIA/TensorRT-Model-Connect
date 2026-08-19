/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2_hoi/pipeline.h"
#include "runtime/models/sam2_hoi/sam2_hoi_video_session.h"
#include "trtmc/models/sam2_hoi_video.h"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

void test_fixed_abi() {
    static_assert(TRTMC_SAM2_HOI_VIDEO_FRAME_COUNT_V1 == 5U);
    static_assert(sizeof(TrtmcSam2HoiVideoRunResultV1) == 64U);
    static_assert(offsetof(TrtmcSam2HoiVideoRunResultV1, struct_size) == 0U);
    static_assert(offsetof(TrtmcSam2HoiVideoRunResultV1, abi_version) == 8U);
    static_assert(offsetof(TrtmcSam2HoiVideoRunResultV1, produced_frame_count) == 12U);
    static_assert(offsetof(TrtmcSam2HoiVideoRunResultV1, reserved_u64) == 16U);
    static_assert(std::extent_v<decltype(TrtmcSam2HoiVideoRunResultV1::reserved_u64)> == 6U);
    check(trtmc_sam2_hoi_video_abi_version() == TRTMC_SAM2_HOI_VIDEO_ABI_VERSION_1,
          "SAM2 HOI exposes ABI version 1");
}

void test_boundary_failures() {
    TrtmcSam2HoiVideoRunResultV1 result{};
    auto status =
        trtmc_sam2_hoi_video_run_jpeg_files_v1(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, &result, sizeof(result) - 1U);
    check(status == TRTMC_SAM2_HOI_VIDEO_STATUS_UNSUPPORTED_ABI,
          "undersized result is rejected before use");
    check(std::string(trtmc_sam2_hoi_video_last_error()).find("too small") != std::string::npos,
          "undersized result reports a diagnostic");

    status =
        trtmc_sam2_hoi_video_run_jpeg_files_v1(nullptr, "0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg",
                                               "tracking.json", "", &result, sizeof(result));
    check(status == TRTMC_SAM2_HOI_VIDEO_STATUS_INVALID_ARGUMENT,
          "exactly one empty output path is rejected during preflight");
    check(std::string(trtmc_sam2_hoi_video_last_error()).find("both be empty") != std::string::npos,
          "partial output request reports a diagnostic");

    status = trtmc_sam2_hoi_video_run_jpeg_files_v1(nullptr, "0.jpg", "1.jpg", "2.jpg", "3.jpg",
                                                    "4.jpg", "masks/frame_000000.npy", "masks",
                                                    &result, sizeof(result));
    check(status == TRTMC_SAM2_HOI_VIDEO_STATUS_INVALID_ARGUMENT,
          "output collision is rejected before session processing");
    check(std::string(trtmc_sam2_hoi_video_last_error()).find("generated mask") !=
              std::string::npos,
          "output collision reports a diagnostic");

    status = trtmc_sam2_hoi_video_run_jpeg_files_v1(nullptr, "0.jpg", "1.jpg", "2.png", "3.jpg",
                                                    "4.jpg", "", "", &result, sizeof(result));
    check(status == TRTMC_SAM2_HOI_VIDEO_STATUS_INVALID_ARGUMENT,
          "non-JPEG input is rejected during preflight");

    status = trtmc_sam2_hoi_video_run_jpeg_files_v1(nullptr, "0.jpg", "1.jpg", "2.jpg", "3.jpg",
                                                    "4.jpg", "", "", &result, sizeof(result));
    check(status == TRTMC_SAM2_HOI_VIDEO_STATUS_INVALID_ARGUMENT, "null session is rejected");
    check(std::string(trtmc_sam2_hoi_video_last_error()).find("null") != std::string::npos,
          "null session reports a diagnostic");
    trtmc_sam2_hoi_video_session_destroy(nullptr);
}

void test_factory_preflight_and_internal_ownership_guard() {
    auto* session = trtmc_sam2_hoi_video_create_from_bundle_v1(nullptr, "plugin", "backend");
    check(session == nullptr, "SAM2 HOI C API rejects a missing bundle path");
    check(std::string(trtmc_sam2_hoi_video_last_error()).find("paths are required") !=
              std::string::npos,
          "SAM2 HOI factory preflight reports a diagnostic");

    session = trtmc_sam2_hoi_video_create_from_bundle_v1("/missing-user-built.bundle", ".", ".");
    check(session == nullptr, "SAM2 HOI C API enters the bundle loader");
    check(std::string(trtmc_sam2_hoi_video_last_error()).find("Failed to open bundle file") !=
              std::string::npos,
          "SAM2 HOI missing bundle failure crosses the C boundary");

    bool rejected_missing_pipeline = false;
    try {
        (void)trtmc::sam2_hoi::makeVideoSessionHandle(nullptr);
    } catch (const std::invalid_argument&) {
        rejected_missing_pipeline = true;
    }
    check(rejected_missing_pipeline, "session handle cannot own a null pipeline");
}

} // namespace

int main() {
    test_fixed_abi();
    test_boundary_failures();
    test_factory_preflight_and_internal_ownership_guard();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
