/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_CONTROL_H
#define TRTMC_CONTROL_H

#include "trtmc/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trtmc_bundle trtmc_bundle;

typedef struct {
    const uint8_t* data;
    uint64_t size;
} trtmc_bytes_view;

typedef struct {
    trtmc_string_view name;
    uint64_t offset;
    uint64_t length;
} trtmc_bundle_section_v1;

/* Strings and sections are borrowed until bundle_release. Opening a bundle
 * does not load its backend or model family. Section offsets are relative to
 * the payload region; callers normally read sections by name. */
typedef struct {
    int32_t format;
    trtmc_string_view family;
    trtmc_string_view task;
    trtmc_string_view backend;
    const trtmc_bundle_section_v1* sections;
    uint64_t section_count;
} trtmc_bundle_info_v1;

typedef struct {
    uint64_t struct_size;
    trtmc_string_view runtime_root; /* Empty selects the C ABI library's directory. */
    trtmc_string_view library;
    trtmc_string_view function;
    trtmc_string_view kernel_name;
} trtmc_byok_options_v1;

/* Model-local IDs and loading/replacement policy belong to the family. A task
 * selects an already loaded adapter through its declared Config key; no
 * model-global active-adapter setting is introduced. list returns an owned
 * result snapshot, released through core.result_release. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* load)(trtmc_model*, trtmc_string_view adapter_id,
                                   trtmc_string_view adapter_path, trtmc_error**);
    trtmc_status(TRTMC_CALL* unload)(trtmc_model*, trtmc_string_view adapter_id, trtmc_error**);
    trtmc_status(TRTMC_CALL* list)(trtmc_model*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* list_view)(const trtmc_result*, trtmc_strings_view*, trtmc_error**);
} trtmc_lora_api_v1;

#ifdef __cplusplus
}
#endif

#endif /* TRTMC_CONTROL_H */
