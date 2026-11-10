/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/siglip/runtime/tokenizer.h"
#include <nlohmann/json.hpp>

namespace trtmc {
std::unique_ptr<ITokenizer> CreateSiglipBpeTokenizer(const nlohmann::json& tokenizer,
                                                  bool add_special_tokens);
} // namespace trtmc
