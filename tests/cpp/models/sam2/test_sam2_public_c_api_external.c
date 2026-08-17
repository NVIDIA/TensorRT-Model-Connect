/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <trtmc/models/sam2_video.h>

_Static_assert(TRTMC_SAM2_VIDEO_ABI_VERSION_1 == 1U, "unexpected SAM2 ABI version");
_Static_assert(TRTMC_SAM2_VIDEO_FRAME_COUNT_V1 == 5U, "unexpected SAM2 frame count");
_Static_assert(sizeof(TrtmcSam2VideoTrackV1) > 0U, "track ABI must be complete");
_Static_assert(sizeof(TrtmcSam2VideoFrameResultV1) > 0U, "frame ABI must be complete");

int sam2_public_c_api_external_consumer(void) {
    TrtmcSam2VideoSession* session = (TrtmcSam2VideoSession*)0;
    TrtmcSam2VideoTrackV1 track = {0};
    uint32_t (*abi_version)(void) = trtmc_sam2_video_abi_version;
    TrtmcSam2VideoSession* (*create)(const char*, const char*, const char*, const char*) =
        trtmc_sam2_video_create_from_qualified_bundle_v1;
    void (*destroy)(TrtmcSam2VideoSession*) = trtmc_sam2_video_session_destroy;
    track.struct_size = sizeof(track);
    track.abi_version = TRTMC_SAM2_VIDEO_ABI_VERSION_1;
    (void)abi_version;
    (void)create;
    (void)destroy;
    return session == (TrtmcSam2VideoSession*)0 &&
                   track.abi_version == TRTMC_SAM2_VIDEO_ABI_VERSION_1
               ? TRTMC_SAM2_VIDEO_STATUS_OK
               : TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI;
}
