/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_TYPES_H
#define TRTMC_TYPES_H

#include <stdint.h>

#if defined(_WIN32)
#define TRTMC_CALL __cdecl
#if defined(TRTMC_C_BUILD)
#define TRTMC_EXPORT __declspec(dllexport)
#else
#define TRTMC_EXPORT __declspec(dllimport)
#endif
#else
#define TRTMC_CALL
#define TRTMC_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Experimental until the SDK's first stable release. Element layouts are
 * fixed within a major version, including elements used in arrays. */
typedef int32_t trtmc_status;
enum {
    TRTMC_OK = 0,
    TRTMC_END = 1,
    TRTMC_INVALID_ARGUMENT = 2,
    TRTMC_INVALID_CONFIG = 3,
    TRTMC_UNSUPPORTED = 4,
    TRTMC_VERSION_MISMATCH = 5,
    TRTMC_OUT_OF_MEMORY = 6,
    TRTMC_INTERNAL_ERROR = 7,
    TRTMC_AGAIN = 8, /* A bounded stream poll timed out without an event. */
    TRTMC_BUSY = 9   /* A live session owns this model's execution state. */
};

typedef struct trtmc_model trtmc_model;
typedef struct trtmc_result trtmc_result;
typedef struct trtmc_error trtmc_error;

typedef struct {
    const char* data;
    uint64_t size;
} trtmc_string_view;

typedef struct {
    const int32_t* data;
    uint64_t size;
} trtmc_i32_view;

typedef struct {
    const int64_t* data;
    uint64_t size;
} trtmc_i64_view;

typedef struct {
    const double* data;
    uint64_t size;
} trtmc_f64_view;

/* Borrowed host float32 data, not a Config value kind. */
typedef struct {
    const float* data;
    uint64_t size;
} trtmc_f32_view;

typedef struct {
    const trtmc_string_view* data;
    uint64_t size;
} trtmc_strings_view;

enum {
    TRTMC_CONFIG_I64 = 1,
    TRTMC_CONFIG_F64 = 2,
    TRTMC_CONFIG_BOOL = 3,
    TRTMC_CONFIG_STRING = 4,
    TRTMC_CONFIG_I64_LIST = 5,
    TRTMC_CONFIG_F64_LIST = 6,
    TRTMC_CONFIG_STRING_LIST = 7
};

typedef struct {
    uint32_t kind;
    union {
        int64_t i64;
        double f64;
        uint32_t boolean; /* Exactly zero or one. */
        trtmc_string_view string;
        trtmc_i64_view i64_list;
        trtmc_f64_view f64_list;
        trtmc_strings_view string_list;
    } as;
} trtmc_config_value_v1;

typedef struct {
    trtmc_string_view name;
    trtmc_config_value_v1 value;
} trtmc_config_entry_v1;

typedef struct {
    const trtmc_config_entry_v1* entries;
    uint64_t count;
} trtmc_config_view_v1;

typedef struct {
    trtmc_string_view name;
    uint32_t kind;
    uint32_t has_fixed_default;
    trtmc_config_value_v1 default_value;
    trtmc_string_view description;
} trtmc_config_field_v1;

typedef struct {
    uint32_t major;
    uint32_t minor;
    uint64_t byte_size;
} trtmc_api_header;

typedef struct {
    uint64_t struct_size;
    trtmc_string_view runtime_root;
    uint64_t kv_cache_size_bytes;
    trtmc_string_view runtime_cache_path;
    uint32_t cuda_graphs;
} trtmc_load_options_v1;

typedef struct {
    trtmc_string_view family;
    trtmc_string_view backend;
    trtmc_string_view bundle_task;
} trtmc_model_info_v1;

typedef struct {
    trtmc_string_view id;
    uint32_t major;
    uint32_t minor;
} trtmc_task_info_v1;

enum { TRTMC_TEXT_UTF8 = 1, TRTMC_TEXT_TOKEN_IDS = 2 };

typedef struct {
    uint32_t kind;
    union {
        trtmc_string_view text;
        trtmc_i32_view token_ids;
    } as;
} trtmc_text_source_v1;

typedef struct {
    trtmc_text_source_v1 prefix;
} trtmc_text_continuation_request_v1;

typedef struct {
    double start_seconds;
    double end_seconds;
    trtmc_string_view text;
    trtmc_i32_view token_ids;
} trtmc_transcription_segment_v1;

typedef struct {
    trtmc_string_view text;
    trtmc_i32_view token_ids;
    double setup_ms;
    double prefill_ms;
    double decode_ms;
    const trtmc_transcription_segment_v1* segments;
    uint64_t segment_count;
} trtmc_text_result_view_v1;

#ifdef __cplusplus
}
#endif

#endif /* TRTMC_TYPES_H */
