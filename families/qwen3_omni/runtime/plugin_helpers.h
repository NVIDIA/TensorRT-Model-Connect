/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/qwen3_omni/runtime/tokenizer.h"
#include "trtmc/bundle.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::qwen3_omni {

std::vector<char> require_section(const BundleReader& bundle, std::string_view name);
std::string require_text_section(const BundleReader& bundle, std::string_view name);
std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle);

} // namespace trtmc::qwen3_omni
