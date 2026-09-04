/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/lfm2/runtime/kv_cache.h"
#include "families/lfm2/runtime/pipeline.h"
#include "families/lfm2/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::lfm2 {

std::vector<char> require_section(const BundleReader& bundle, std::string_view name);
std::string require_text_section(const BundleReader& bundle, std::string_view name);
std::unique_ptr<ITrtModule> load_module(IBackend& backend, const std::vector<char>& plan);
std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle);
DType state_dtype(const std::string& precision);
Lfm2KvCacheNames kv_names(std::int32_t num_attention_layers);
void apply_chat_template(const BundleReader& bundle, Lfm2TextGenConfig& config);

} // namespace trtmc::lfm2
