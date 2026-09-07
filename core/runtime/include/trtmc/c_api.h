/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_C_API_H
#define TRTMC_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trtmc_task trtmc_task_t;

typedef enum trtmc_status {
    TRTMC_STATUS_OK = 0,
    TRTMC_STATUS_INVALID_ARGUMENT = 1,
    TRTMC_STATUS_WRONG_TASK = 2,
    TRTMC_STATUS_RUNTIME_ERROR = 3,
    TRTMC_STATUS_OUT_OF_MEMORY = 4
} trtmc_status_t;

typedef struct trtmc_text_result {
    char* text;
    size_t text_length;
    int32_t* token_ids;
    size_t token_count;
    double setup_ms;
    double prefill_ms;
    double decode_ms;
} trtmc_text_result_t;

typedef struct trtmc_image_result {
    float* pixels;
    size_t pixel_count;
    int32_t height;
    int32_t width;
    int32_t channels;
    int32_t num_frames;
} trtmc_image_result_t;

/*
 * Return the current library version. The returned process-lifetime string is
 * owned by TensorRT-Model-Connect and must not be freed.
 */
const char* trtmc_version(void);

/*
 * Return the error for the most recent failed status-returning API call on this
 * thread. A successful status-returning call clears it. The returned pointer is
 * valid until the next such call on the same thread and must not be freed.
 */
const char* trtmc_last_error(void);

/*
 * Load exactly the family and backend named by bundle_path from runtime_root.
 * Both paths and out_task must be non-null and non-empty where applicable.
 * On success, *out_task is caller-owned and must be passed to
 * trtmc_task_destroy. On failure, *out_task is NULL.
 */
trtmc_status_t trtmc_task_load(const char* bundle_path, const char* runtime_root,
                               trtmc_task_t** out_task);

/* Destroy a task handle. Passing NULL is allowed. */
void trtmc_task_destroy(trtmc_task_t* task);

/*
 * Return the task identity declared by the loaded implementation. The borrowed
 * string is valid until task is destroyed and must not be freed.
 */
trtmc_status_t trtmc_task_name(const trtmc_task_t* task, const char** out_name);

/*
 * Invoke an ITextGeneration task. prompt and out_result must be non-null and
 * max_new_tokens must be positive. out_result must be zero-initialized or have
 * been released already. On success its text and token_ids are caller-owned and
 * must be released with trtmc_text_result_free. On failure it is zeroed.
 */
trtmc_status_t trtmc_text_generate(trtmc_task_t* task, const char* prompt, int32_t max_new_tokens,
                                   trtmc_text_result_t* out_result);

/* Release one text result and zero all of its fields. Passing NULL is allowed. */
void trtmc_text_result_free(trtmc_text_result_t* result);

/*
 * Invoke an IImageBatchGeneration task. prompts, seeds, and out_results each
 * contain count entries; every prompt must be non-null. count and num_steps must
 * be positive, and guidance_scale must be finite and non-negative. Each output
 * element must be zero-initialized or have been released already. On success
 * every pixels buffer is caller-owned and must be released separately with
 * trtmc_image_result_free. On failure every output element is zeroed.
 */
trtmc_status_t trtmc_image_generate_batch(trtmc_task_t* task, const char* const* prompts,
                                          const uint32_t* seeds, size_t count, int32_t num_steps,
                                          float guidance_scale, trtmc_image_result_t* out_results);

/* Release one image result and zero all of its fields. Passing NULL is allowed. */
void trtmc_image_result_free(trtmc_image_result_t* result);

#ifdef __cplusplus
}
#endif

#endif
