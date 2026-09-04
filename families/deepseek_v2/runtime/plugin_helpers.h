/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/deepseek_v2/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::deepseek_v2 {

std::vector<char> require_section(const BundleReader& bundle, std::string_view name);
std::string require_text_section(const BundleReader& bundle, std::string_view name);
std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* label);
std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle);
std::int32_t rank_local_kv_dim(std::int32_t num_key_value_heads, std::int32_t head_dim,
                               std::int32_t tensor_parallel_size);

} // namespace trtmc::deepseek_v2
