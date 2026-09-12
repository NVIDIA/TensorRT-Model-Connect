/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_TOOLS_H
#define TRTMC_TOOLS_H

#include "trtmc/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Explicit tool payloads, not a generic serialized Task. No tool executes
 * automatically. call_id identifies the request/result relationship; speech
 * sessions additionally carry their epoch in the session/event envelope. */
typedef struct {
    trtmc_string_view name;
    trtmc_string_view description;
    trtmc_string_view parameters_schema_json;
} trtmc_tool_definition_v1;
typedef struct {
    trtmc_string_view call_id;
    trtmc_string_view name;
    trtmc_string_view arguments_json;
    uint32_t state; /* Family-provided state; populated fields do not imply Complete. */
} trtmc_tool_call_v1;
enum {
    TRTMC_TOOL_CALL_UNKNOWN = 0,
    TRTMC_TOOL_CALL_COMPLETE = 1,
    TRTMC_TOOL_CALL_INCOMPLETE = 2,
    TRTMC_TOOL_CALL_MALFORMED = 3
};
typedef struct {
    trtmc_string_view call_id;
    trtmc_string_view content_text;
    uint32_t is_error;
} trtmc_tool_result_v1;

#ifdef __cplusplus
}
#endif
#endif /* TRTMC_TOOLS_H */
