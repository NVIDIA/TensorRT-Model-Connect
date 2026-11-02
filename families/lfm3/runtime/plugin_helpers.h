/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/lfm3/runtime/kv_cache.h"
#include "families/lfm3/runtime/pipeline.h"
#include "families/lfm3/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::lfm3 {

std::vector<char> require_section(const BundleReader& bundle, std::string_view name);
std::string require_text_section(const BundleReader& bundle, std::string_view name);
std::unique_ptr<ITrtModule> load_module(IBackend& backend, const std::vector<char>& plan);
std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle);
DType state_dtype(const std::string& precision);
Lfm3KvCacheNames kv_names(std::int32_t num_attention_layers);
void apply_chat_template(const BundleReader& bundle, Lfm3TextGenConfig& config);

} // namespace trtmc::lfm3
