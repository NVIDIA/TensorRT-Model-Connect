/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/tokenizer.h"

#include <cstddef>
#include <memory>

namespace trtmc {

std::unique_ptr<ITokenizer> CreateSanaWmBpeTokenizer(const char* tokenizer_json_data,
                                                     std::size_t tokenizer_json_size,
                                                     bool add_special_tokens);

} // namespace trtmc
