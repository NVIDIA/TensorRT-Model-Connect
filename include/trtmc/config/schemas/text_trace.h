/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Schema for the "text_trace" namespace. Mirrors
// python/tensorrt_model_connect/runtime_config/schemas/text_trace.py one-for-one.

#include "trtmc/config/schema_registry.h"

namespace trtmc::config::schemas {

Schema make_text_trace_schema();

} // namespace trtmc::config::schemas
