/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/qwen/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::qwen {

std::vector<char> require_section(const BundleReader& bundle, std::string_view name);
std::string require_text_section(const BundleReader& bundle, std::string_view name);
std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* label);
std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle);
void validate_kv_row_contract(const ITrtModule& module, bool runtime_sized, std::int32_t num_layers,
                              std::int32_t kv_dim, std::int32_t bundle_max_rows, const char* label);
std::int32_t runtime_cache_rows(std::uint64_t requested_bytes, std::int32_t bundle_max_rows,
                                std::int32_t num_layers, std::int32_t num_key_value_heads,
                                std::int32_t head_dim, DType cache_dtype);

} // namespace trtmc::qwen
