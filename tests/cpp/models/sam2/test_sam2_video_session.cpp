/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/models/sam2_video.h"

#include <cstddef>
#include <cstdlib>
#include <iostream>
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
    static_assert(TRTMC_SAM2_VIDEO_FRAME_COUNT_V1 == 5U);
    static_assert(TRTMC_SAM2_VIDEO_FRAME_HEIGHT_V1 == 1280);
    static_assert(TRTMC_SAM2_VIDEO_FRAME_WIDTH_V1 == 1088);
    static_assert(std::extent_v<decltype(TrtmcSam2VideoRunResultV1::masks)> == 5U);
    static_assert(sizeof(TrtmcSam2VideoRunResultV1) == 120U);
    static_assert(offsetof(TrtmcSam2VideoRunResultV1, masks) == 48U);
    static_assert(offsetof(TrtmcSam2VideoRunResultV1, reserved_u64) == 88U);
    check(trtmc_sam2_video_abi_version() == TRTMC_SAM2_VIDEO_ABI_VERSION_1,
          "SAM2 exposes ABI version 1");
    check(TRTMC_SAM2_VIDEO_RUN_DEFAULT == 0 &&
              TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE != TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST,
          "default and materialized mask ownership are explicit");
}

void test_boundary_failures() {
    TrtmcSam2VideoRunResultV1 result{};
    auto status =
        trtmc_sam2_video_run_rgb8_v1(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                     TRTMC_SAM2_VIDEO_RUN_DEFAULT, &result, sizeof(result) - 1U);
    check(status == TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI,
          "undersized result is rejected before use");
    check(std::string(trtmc_sam2_video_last_error()).find("too small") != std::string::npos,
          "undersized result reports a diagnostic");

    status = trtmc_sam2_video_run_rgb8_v1(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                          UINT32_C(0x80000000), &result, sizeof(result));
    check(status == TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI, "unknown run flags are rejected");

    status = trtmc_sam2_video_run_rgb8_v1(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                          TRTMC_SAM2_VIDEO_RUN_DEFAULT, &result, sizeof(result));
    check(status == TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT, "null session is rejected");
    check(std::string(trtmc_sam2_video_last_error()).find("null") != std::string::npos,
          "null session reports a diagnostic");
    trtmc_sam2_video_session_destroy(nullptr);
}

} // namespace

int main() {
    test_fixed_abi();
    test_boundary_failures();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
