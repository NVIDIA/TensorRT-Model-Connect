/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_C_API_H
#define TRTMC_C_API_H

#include "trtmc/action.h"
#include "trtmc/audio.h"
#include "trtmc/control.h"
#include "trtmc/features.h"
#include "trtmc/image.h"
#include "trtmc/language.h"
#include "trtmc/numeric.h"
#include "trtmc/perception.h"
#include "trtmc/recurrent.h"
#include "trtmc/speech.h"
#include "trtmc/stream.h"
#include "trtmc/tracking.h"
#include "trtmc/types.h"
#include "trtmc/video.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TRTMC_TASK_TEXT_CONTINUATION "text_continuation"

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_continuation_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_continuation_api_v1;

/* Each function initializes its output before work. Failure status is
 * authoritative even when allocation failure prevents an error detail.
 * Null result/model/error handles may be passed to their release functions.
 * Borrowed metadata lives until model_release; result views until result_release.
 * A Task table is static but every invocation checks the actual model. */
typedef struct {
    trtmc_api_header header;
    trtmc_string_view(TRTMC_CALL* runtime_version)(void);
    trtmc_status(TRTMC_CALL* error_code)(const trtmc_error*);
    trtmc_string_view(TRTMC_CALL* error_message)(const trtmc_error*);
    void(TRTMC_CALL* error_release)(trtmc_error*);
    trtmc_status(TRTMC_CALL* model_load)(trtmc_string_view bundle_path,
                                         const trtmc_load_options_v1*, trtmc_model**,
                                         trtmc_error**);
    void(TRTMC_CALL* model_release)(trtmc_model*);
    trtmc_status(TRTMC_CALL* model_info)(const trtmc_model*, trtmc_model_info_v1*, trtmc_error**);
    trtmc_status(TRTMC_CALL* model_task_count)(const trtmc_model*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* model_task_info)(const trtmc_model*, uint64_t index,
                                              trtmc_task_info_v1*, trtmc_error**);
    trtmc_status(TRTMC_CALL* model_get_task_api)(const trtmc_model*, trtmc_string_view task,
                                                 uint32_t major, uint32_t minor,
                                                 const trtmc_api_header**, trtmc_error**);
    trtmc_status(TRTMC_CALL* config_field_count)(const trtmc_model*, trtmc_string_view task,
                                                 uint32_t major, uint32_t minor, uint64_t*,
                                                 trtmc_error**);
    trtmc_status(TRTMC_CALL* config_field_info)(const trtmc_model*, trtmc_string_view task,
                                                uint32_t major, uint32_t minor, uint64_t index,
                                                trtmc_config_field_v1*, trtmc_error**);
    void(TRTMC_CALL* result_release)(trtmc_result*);
    trtmc_status(TRTMC_CALL* bundle_open)(trtmc_string_view path, trtmc_bundle**, trtmc_error**);
    void(TRTMC_CALL* bundle_release)(trtmc_bundle*);
    trtmc_status(TRTMC_CALL* bundle_info)(const trtmc_bundle*, trtmc_bundle_info_v1*,
                                          trtmc_error**);
    trtmc_status(TRTMC_CALL* bundle_read_section)(const trtmc_bundle*, trtmc_string_view name,
                                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* bytes_result_view)(const trtmc_result*, trtmc_bytes_view*,
                                                trtmc_error**);
    trtmc_status(TRTMC_CALL* byok_load)(const trtmc_byok_options_v1*, trtmc_error**);
    trtmc_status(TRTMC_CALL* model_get_lora_api)(const trtmc_model*, uint32_t major, uint32_t minor,
                                                 const trtmc_lora_api_v1**, trtmc_error**);
} trtmc_core_api_v1;

/* The only exported SDK symbol. A successful result always reports the
 * requested table version; unsupported major/minor returns VERSION_MISMATCH. */
TRTMC_EXPORT trtmc_status TRTMC_CALL trtmc_get_api(uint32_t major, uint32_t minor,
                                                   const trtmc_core_api_v1** out_api);

#ifdef __cplusplus
}
#endif

#endif /* TRTMC_C_API_H */
