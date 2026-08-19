/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

/* Fixed model-owned SAM2-HOI five-JPEG video ABI. */
#define TRTMC_SAM2_HOI_VIDEO_ABI_VERSION_1 UINT32_C(1)
#define TRTMC_SAM2_HOI_VIDEO_FRAME_COUNT_V1 UINT32_C(5)

typedef struct TrtmcSam2HoiVideoSession TrtmcSam2HoiVideoSession;

typedef enum TrtmcSam2HoiVideoStatus {
    TRTMC_SAM2_HOI_VIDEO_STATUS_OK = 0,
    TRTMC_SAM2_HOI_VIDEO_STATUS_INVALID_ARGUMENT = -1,
    TRTMC_SAM2_HOI_VIDEO_STATUS_INVALID_STATE = -2,
    TRTMC_SAM2_HOI_VIDEO_STATUS_RUNTIME_ERROR = -3,
    TRTMC_SAM2_HOI_VIDEO_STATUS_UNSUPPORTED_ABI = -4
} TrtmcSam2HoiVideoStatus;

/*
 * Scalar result for one fixed five-frame run. No output allocation crosses
 * the ABI. When materialization is requested, JSON and masks are owned by the
 * caller-selected filesystem paths.
 */
typedef struct TrtmcSam2HoiVideoRunResultV1 {
    uint64_t struct_size;
    uint32_t abi_version;
    int32_t produced_frame_count;
    uint64_t reserved_u64[6];
} TrtmcSam2HoiVideoRunResultV1;

#ifdef __cplusplus
extern "C" {
#define TRTMC_SAM2_HOI_VIDEO_NOEXCEPT noexcept
#else
#define TRTMC_SAM2_HOI_VIDEO_NOEXCEPT
#endif

uint32_t trtmc_sam2_hoi_video_abi_version(void) TRTMC_SAM2_HOI_VIDEO_NOEXCEPT;
const char* trtmc_sam2_hoi_video_last_error(void) TRTMC_SAM2_HOI_VIDEO_NOEXCEPT;

/* The caller rebuilds and owns the complete local bundle. */
TrtmcSam2HoiVideoSession*
trtmc_sam2_hoi_video_create_from_bundle_v1(const char* bundle_path, const char* plugin_dir,
                                           const char* backend_dir) TRTMC_SAM2_HOI_VIDEO_NOEXCEPT;
void trtmc_sam2_hoi_video_session_destroy(TrtmcSam2HoiVideoSession* session)
    TRTMC_SAM2_HOI_VIDEO_NOEXCEPT;

/*
 * Decode and track exactly five JPEG files in the supplied temporal order.
 * Calls on one session must be serialized. Empty output_json and
 * output_masks_dir select the benchmark discard path; both nonempty strings
 * materialize the existing JSON and NumPy-mask contract. Exactly one empty
 * output path is invalid.
 *
 * A successful call replaces the prior scalar result and reuses the runtime
 * modules. A failure after JPEG processing begins poisons the session;
 * argument and ABI preflight failures do not.
 */
int32_t trtmc_sam2_hoi_video_run_jpeg_files_v1(
    TrtmcSam2HoiVideoSession* session, const char* frame0, const char* frame1, const char* frame2,
    const char* frame3, const char* frame4, const char* output_json, const char* output_masks_dir,
    TrtmcSam2HoiVideoRunResultV1* result,
    uint64_t result_struct_size) TRTMC_SAM2_HOI_VIDEO_NOEXCEPT;

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef TRTMC_SAM2_HOI_VIDEO_NOEXCEPT
