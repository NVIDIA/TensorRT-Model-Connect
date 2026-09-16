/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stddef.h>

// Internal, release-coupled CLI entry point. This does not extend the public
// Task ABI. Values are a JSON object parsed from the family-owned declaration.
// All inputs and callbacks borrow the synchronous call's lifetime. Family
// implementations catch exceptions and report errors through the error sink.
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*trtmc_cli_write_v1)(void* context, const char* data, size_t size);
typedef int (*trtmc_family_cli_fn_v1)(const char* handler, const char* values_json,
                                      const char* default_runtime_root, void* context,
                                      trtmc_cli_write_v1 output, trtmc_cli_write_v1 error);

int trtmc_family_cli_v1(const char* handler, const char* values_json,
                        const char* default_runtime_root, void* context, trtmc_cli_write_v1 output,
                        trtmc_cli_write_v1 error);

#ifdef __cplusplus
}
#endif
