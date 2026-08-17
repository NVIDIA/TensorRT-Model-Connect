/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

/*
 * Public model-owned SAM2 bbox-video ABI.
 *
 * Version 1 deliberately exposes only the workload implemented by the model
 * graphs: five contiguous RGB frames and one object selected by the detector
 * on frame zero. The detector label, confidence, and original-space prompt box
 * are track metadata; propagated frame results contain only the singular mask
 * for that track. In particular, this ABI does not claim that the tracker
 * produces a box or confidence for frames one through four.
 *
 * The session owns every mask pointer returned in
 * TrtmcSam2VideoFrameResultV1 until reset or destroy, including a frame-zero
 * prompt view obtained before propagation. Calls operating on the same session
 * must be serialized by the caller.
 */

#define TRTMC_SAM2_VIDEO_ABI_VERSION_1 UINT32_C(1)
#define TRTMC_SAM2_VIDEO_FRAME_COUNT_V1 UINT32_C(5)
#define TRTMC_SAM2_VIDEO_OBJECT_COUNT_V1 UINT32_C(1)

typedef struct TrtmcSam2VideoSession TrtmcSam2VideoSession;

typedef enum TrtmcSam2VideoStatus {
    TRTMC_SAM2_VIDEO_STATUS_OK = 0,
    TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT = -1,
    TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE = -2,
    TRTMC_SAM2_VIDEO_STATUS_PROCESSOR_ERROR = -3,
    TRTMC_SAM2_VIDEO_STATUS_INVALID_RESULT = -4,
    TRTMC_SAM2_VIDEO_STATUS_OUT_OF_RANGE = -5,
    TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI = -6,
    TRTMC_SAM2_VIDEO_STATUS_OVERFLOW = -7,
    TRTMC_SAM2_VIDEO_STATUS_INTERNAL_ERROR = -8
} TrtmcSam2VideoStatus;

typedef enum TrtmcSam2VideoMaskMemoryKind {
    TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST = 1,
    TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE = 2
} TrtmcSam2VideoMaskMemoryKind;

typedef enum TrtmcSam2VideoTrackFlags {
    TRTMC_SAM2_VIDEO_TRACK_PROMPT_BOX_ABSOLUTE_XYXY = UINT32_C(1) << 0
} TrtmcSam2VideoTrackFlags;

typedef enum TrtmcSam2VideoFrameFlags {
    TRTMC_SAM2_VIDEO_FRAME_MASK_BINARY = UINT32_C(1) << 0
} TrtmcSam2VideoFrameFlags;

typedef enum TrtmcSam2VideoGetResultFlags {
    TRTMC_SAM2_VIDEO_GET_DEFAULT = 0,
    TRTMC_SAM2_VIDEO_GET_MATERIALIZE_MASK_HOST = UINT32_C(1) << 0
} TrtmcSam2VideoGetResultFlags;

/*
 * Version-1 selected-track metadata.
 *
 * label and detector_score come from the single frame-zero detection selected
 * as the tracker prompt. prompt_box_xyxy contains four finite, ordered floats
 * in x1,y1,x2,y2 order in the original frame coordinate system. The box is
 * intentionally unclipped and is not a propagated tracker box.
 */
typedef struct TrtmcSam2VideoTrackV1 {
    uint64_t struct_size;
    uint32_t abi_version;
    uint32_t flags;

    int32_t label;
    float detector_score;
    float prompt_box_xyxy[4];

    uint64_t reserved_u64[4];
} TrtmcSam2VideoTrackV1;

/*
 * Version-1 borrowed per-frame mask view for the selected track.
 *
 * frame_index is contiguous in [0, TRTMC_SAM2_VIDEO_FRAME_COUNT_V1). The mask
 * is tightly packed uint8 data in [y][x] order: mask_row_stride_bytes is width
 * and mask_byte_count is height * width.
 *
 * A non-empty CUDA-device mask is valid on mask_device_ordinal. Its pointer is
 * an in-process CUDA Runtime device pointer, not a CUDA IPC handle. The
 * processor must finish all writes to the allocation before its prompt or
 * propagation callback returns; this ABI exposes no producer stream or event,
 * and a result query performs no additional synchronization. Consumers may
 * therefore enqueue work after a successful query on the same process and
 * device primary context. A default query never performs a device-to-host
 * copy. Accuracy callers may request a lazily materialized, session-cached
 * host view.
 */
typedef struct TrtmcSam2VideoFrameResultV1 {
    uint64_t struct_size;
    uint32_t abi_version;
    uint32_t flags;

    int32_t frame_index;
    int32_t height;
    int32_t width;
    int32_t mask_device_ordinal;

    const void* mask;
    uint64_t mask_byte_count;
    uint64_t mask_row_stride_bytes;
    uint32_t mask_memory_kind;
    uint32_t reserved_u32;
    uint64_t reserved_u64[4];
} TrtmcSam2VideoFrameResultV1;

#ifdef __cplusplus
extern "C" {
#define TRTMC_SAM2_VIDEO_NOEXCEPT noexcept
#else
#define TRTMC_SAM2_VIDEO_NOEXCEPT
#endif

uint32_t trtmc_sam2_video_abi_version(void) TRTMC_SAM2_VIDEO_NOEXCEPT;

/*
 * Return the calling thread's most recent SAM2 ABI diagnostic. The borrowed
 * string remains valid until the next SAM2 ABI call on the same thread and
 * may be truncated to keep error reporting allocation-free at the C boundary.
 */
const char* trtmc_sam2_video_last_error(void) TRTMC_SAM2_VIDEO_NOEXCEPT;

/*
 * Legacy constructor retained for ABI compatibility. It always fails closed
 * because it cannot supply the explicit external qualification record required
 * by production admission. Use
 * trtmc_sam2_video_create_from_qualified_bundle_v1 instead.
 */
TrtmcSam2VideoSession*
trtmc_sam2_video_create_from_bundle_v1(const char* bundle_path, const char* plugin_dir,
                                       const char* backend_dir) TRTMC_SAM2_VIDEO_NOEXCEPT;

/*
 * Create a production session from an externally supplied qualification
 * record. The record path is mandatory and is never inferred from the bundle
 * name or a neighboring sidecar. Admission uses only a record whose exact
 * digest is compiled into the SAM2 runtime authority; unpinned records fail
 * before any TensorRT plan module is created.
 */
TrtmcSam2VideoSession* trtmc_sam2_video_create_from_qualified_bundle_v1(
    const char* bundle_path, const char* qualification_record_path, const char* plugin_dir,
    const char* backend_dir) TRTMC_SAM2_VIDEO_NOEXCEPT;

void trtmc_sam2_video_session_destroy(TrtmcSam2VideoSession* session) TRTMC_SAM2_VIDEO_NOEXCEPT;

/*
 * Calling reset invalidates every previously returned mask pointer, even if
 * model-owned cleanup then reports an error. A successful reset clears both
 * wrapper-owned and processor-owned state and makes the session reusable.
 */
int32_t trtmc_sam2_video_reset_v1(TrtmcSam2VideoSession* session) TRTMC_SAM2_VIDEO_NOEXCEPT;

/*
 * A run is strict and one-shot: begin, append exactly five contiguous decoded
 * RGB HWC frames using one append function consistently, run the frame-zero
 * bbox prompt once, then propagate once. The original float32 entrypoint is
 * unchanged: pixels must address height * width * 3 tightly packed floats in
 * [0,1]. The RGB8 entrypoint accepts the decoder-native tightly packed byte
 * representation and enables exact 22-bit Pillow resize on CUDA. Every frame
 * must have identical geometry. The storage and its contents must remain
 * unchanged, with no concurrent writers, until propagate returns or reset
 * succeeds. The normal production path does not add a full CPU validation
 * pass before engine preprocessing; injected diagnostic sessions may opt into
 * that scan.
 *
 * Reuse requires reset. Input frame storage must remain valid until propagate
 * returns or the session is reset. After run_bbox_prompt succeeds, the selected
 * track and frame result zero are queryable. After propagation, result indices
 * [0, 5) expose masks in temporal order; mask pointers from the earlier prompt
 * view remain owned and valid until reset or destroy.
 */
int32_t trtmc_sam2_video_begin_v1(TrtmcSam2VideoSession* session) TRTMC_SAM2_VIDEO_NOEXCEPT;
int32_t trtmc_sam2_video_append_frame_v1(TrtmcSam2VideoSession* session, const float* pixels,
                                         int32_t height, int32_t width) TRTMC_SAM2_VIDEO_NOEXCEPT;
int32_t trtmc_sam2_video_append_frame_rgb8_v1(TrtmcSam2VideoSession* session, const uint8_t* pixels,
                                              int32_t height,
                                              int32_t width) TRTMC_SAM2_VIDEO_NOEXCEPT;
int32_t
trtmc_sam2_video_run_bbox_prompt_v1(TrtmcSam2VideoSession* session) TRTMC_SAM2_VIDEO_NOEXCEPT;
int32_t trtmc_sam2_video_propagate_v1(TrtmcSam2VideoSession* session,
                                      uint64_t* frame_count) TRTMC_SAM2_VIDEO_NOEXCEPT;

int32_t trtmc_sam2_video_get_track_v1(const TrtmcSam2VideoSession* session,
                                      TrtmcSam2VideoTrackV1* track,
                                      uint64_t track_struct_size) TRTMC_SAM2_VIDEO_NOEXCEPT;
int32_t trtmc_sam2_video_result_count_v1(const TrtmcSam2VideoSession* session,
                                         uint64_t* frame_count) TRTMC_SAM2_VIDEO_NOEXCEPT;
int32_t trtmc_sam2_video_get_frame_result_v1(TrtmcSam2VideoSession* session, uint64_t result_index,
                                             uint32_t get_flags,
                                             TrtmcSam2VideoFrameResultV1* result,
                                             uint64_t result_struct_size) TRTMC_SAM2_VIDEO_NOEXCEPT;

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef TRTMC_SAM2_VIDEO_NOEXCEPT
