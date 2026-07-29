/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Small helpers used by the Granite runtime plugin.

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/models/granite/kv_cache.h"
#include "trtmc/runtime/pipeline_plugin.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstddef>
#include <memory>
#include <string>

namespace trtmc {

// Emit a parseable runtime load/deserialization timing line.
void log_trt_load_timing(const char* label, double load_deserialize_ms, std::size_t plan_bytes);

// Create a native tokenizer from bundle. Tries BPE -> WordPiece -> Unigram.
// Returns nullptr if no native tokenizer matches.
std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleFile& bundle);

// Convert the BaseConfig precision string ("fp16", "bf16", "fp32") to a DType
// for use as KV cache element type.
DType cache_dtype_from_precision(const std::string& precision);

} // namespace trtmc
