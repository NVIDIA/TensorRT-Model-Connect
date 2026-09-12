/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_IMAGE_API_H
#define TRTMC_IMAGE_API_H

#include "trtmc/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { TRTMC_IMAGE_UINT8 = 1, TRTMC_IMAGE_FLOAT32 = 2 };

/* Borrowed contiguous host HWC pixels. UInt8 is [0,255], float32 is [0,1].
 * Channels are RGB/RGBA or grayscale. The loaded family validates accepted
 * representations and owns all resizing, normalization and preprocessing. */
typedef struct {
    const void* data;
    uint64_t byte_size;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t format;
} trtmc_image_input_v1;

/* Aligned host HW mask. Zero preserves, one selects generation. */
typedef struct {
    const float* data;
    uint64_t count;
    uint32_t height;
    uint32_t width;
} trtmc_image_mask_v1;

typedef struct {
    trtmc_string_view prompt;
    /* Optional host float32 in the loaded family's documented latent layout.
     * Empty selects family initialization. Family validates size/packing and
     * supported replay/batch combinations; shared code never creates noise. */
    trtmc_f32_view initial_latents;
} trtmc_text_to_image_request_v1;

typedef struct {
    const trtmc_image_input_v1* images;
    uint64_t image_count;
    trtmc_string_view prompt;
    trtmc_f32_view initial_latents;
} trtmc_images_text_to_image_edit_request_v1;

typedef struct {
    trtmc_image_input_v1 source;
    trtmc_image_mask_v1 mask;
    trtmc_string_view prompt;
} trtmc_masked_image_text_to_image_request_v1;

typedef struct {
    trtmc_text_to_image_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_text_to_image_item_v1;

typedef struct {
    const trtmc_batch_text_to_image_item_v1* items;
    uint64_t count;
} trtmc_batch_text_to_image_request_v1;

/* Borrowed from the result owner; one contiguous float32 HWC image.
 * Empty pixels with height=width=0 and channels=3 is a completed non-output
 * distributed participant, not a decoded image. No other empty shape is valid. */
typedef struct {
    const float* pixels;
    uint64_t pixel_count;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
} trtmc_image_result_view_v1;

#define TRTMC_TASK_TEXT_TO_IMAGE "text_to_image"
#define TRTMC_TASK_IMAGES_TEXT_TO_IMAGE_EDIT "images_text_to_image_edit"
#define TRTMC_TASK_MASKED_IMAGE_TEXT_TO_IMAGE "masked_image_text_to_image"
#define TRTMC_TASK_BATCH_TEXT_TO_IMAGE "batch_text_to_image"

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_to_image_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_image_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_to_image_api_v1;

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_images_text_to_image_edit_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_image_result_view_v1*,
                                          trtmc_error**);
} trtmc_images_text_to_image_edit_api_v1;

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_masked_image_text_to_image_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_image_result_view_v1*,
                                          trtmc_error**);
} trtmc_masked_image_text_to_image_api_v1;

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run_batch)(trtmc_model*, const trtmc_batch_text_to_image_request_v1*,
                                        trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item)(const trtmc_result*, uint64_t,
                                          trtmc_image_result_view_v1*, trtmc_error**);
} trtmc_batch_text_to_image_api_v1;

#ifdef __cplusplus
}
#endif

#endif /* TRTMC_IMAGE_API_H */
