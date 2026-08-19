/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

/* Fixed model-owned SAM2 bbox-video ABI. */
#define TRTMC_SAM2_VIDEO_ABI_VERSION_1 UINT32_C(1)
#define TRTMC_SAM2_VIDEO_FRAME_COUNT_V1 UINT32_C(5)
#define TRTMC_SAM2_VIDEO_FRAME_HEIGHT_V1 INT32_C(1280)
#define TRTMC_SAM2_VIDEO_FRAME_WIDTH_V1 INT32_C(1088)

typedef struct TrtmcSam2VideoSession TrtmcSam2VideoSession;

typedef enum TrtmcSam2VideoStatus {
    TRTMC_SAM2_VIDEO_STATUS_OK = 0,
    TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT = -1,
    TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE = -2,
    TRTMC_SAM2_VIDEO_STATUS_RUNTIME_ERROR = -3,
    TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI = -4
} TrtmcSam2VideoStatus;

typedef enum TrtmcSam2VideoMaskMemoryKind {
    TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST = 1,
    TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE = 2
} TrtmcSam2VideoMaskMemoryKind;

typedef enum TrtmcSam2VideoRunFlags {
    /* Fast path: return five session-owned CUDA device pointers. */
    TRTMC_SAM2_VIDEO_RUN_DEFAULT = 0,
    /* Accuracy path: copy all five masks into session-owned host storage. */
    TRTMC_SAM2_VIDEO_RUN_MATERIALIZE_MASKS_HOST = UINT32_C(1) << 0
} TrtmcSam2VideoRunFlags;

/*
 * One result for the fixed five-frame, one-object workload. Masks are tightly
 * packed binary uint8 arrays of FRAME_HEIGHT_V1 by FRAME_WIDTH_V1. All five
 * pointers use mask_memory_kind and mask_device_ordinal. They remain valid
 * until the next run on this session or its destruction.
 */
typedef struct TrtmcSam2VideoRunResultV1 {
    uint64_t struct_size;
    uint32_t abi_version;
    uint32_t mask_memory_kind;
    int32_t mask_device_ordinal;
    int32_t label;
    float detector_score;
    float prompt_box_xyxy[4];
    const void* masks[TRTMC_SAM2_VIDEO_FRAME_COUNT_V1];
    uint64_t reserved_u64[4];
} TrtmcSam2VideoRunResultV1;

#ifdef __cplusplus
extern "C" {
#define TRTMC_SAM2_VIDEO_NOEXCEPT noexcept
#else
#define TRTMC_SAM2_VIDEO_NOEXCEPT
#endif

uint32_t trtmc_sam2_video_abi_version(void) TRTMC_SAM2_VIDEO_NOEXCEPT;
const char* trtmc_sam2_video_last_error(void) TRTMC_SAM2_VIDEO_NOEXCEPT;

/* The caller rebuilds and owns the complete local bundle. */
TrtmcSam2VideoSession*
trtmc_sam2_video_create_from_bundle_v1(const char* bundle_path, const char* plugin_dir,
                                       const char* backend_dir) TRTMC_SAM2_VIDEO_NOEXCEPT;
void trtmc_sam2_video_session_destroy(TrtmcSam2VideoSession* session) TRTMC_SAM2_VIDEO_NOEXCEPT;

/*
 * Run exactly five host-resident 1088x1280 (width x height) tightly packed
 * RGB8 HWC frames.
 * Calls on one session must be serialized. A successful call replaces prior
 * results and reuses the native execution contexts. A failure after processing
 * begins poisons the session; ABI and flag preflight failures do not.
 */
int32_t trtmc_sam2_video_run_rgb8_v1(TrtmcSam2VideoSession* session, const uint8_t* frame0,
                                     const uint8_t* frame1, const uint8_t* frame2,
                                     const uint8_t* frame3, const uint8_t* frame4, uint32_t flags,
                                     TrtmcSam2VideoRunResultV1* result,
                                     uint64_t result_struct_size) TRTMC_SAM2_VIDEO_NOEXCEPT;

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef TRTMC_SAM2_VIDEO_NOEXCEPT
