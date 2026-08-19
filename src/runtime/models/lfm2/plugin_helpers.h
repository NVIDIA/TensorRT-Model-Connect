/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "bundle/bundle_format.h"
#include "runtime/models/lfm2/kv_cache.h"
#include "runtime/models/lfm2/pipeline.h"
#include "trtmc/runtime/pipeline_plugin.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"

#include <memory>
#include <string>

namespace trtmc {

std::unique_ptr<ITrtModule> load_lfm2_module(IBackend* backend, const std::vector<char>* plan,
                                             const ModuleCreateOptions& options = {});

std::shared_ptr<ITokenizer> create_lfm2_tokenizer(const BundleFile& bundle);
DType lfm2_state_dtype(const std::string& precision);
std::string lfm2_expand_layer_name(const std::string& pattern, int32_t layer);
Lfm2KvCacheNames lfm2_kv_names(const BaseConfig& config, int32_t num_attention_layers);
void apply_lfm2_chat_template_format(const BundleFile& bundle, Lfm2TextGenConfig& config);

} // namespace trtmc
