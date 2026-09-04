/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>

namespace trtmc::wan2_2 {

// Reproduces the normalization used by Wan2.2's UMT5 tokenizer for the model's
// supported prompt contract: ftfy character-width repair, two HTML-unescape
// passes, Unicode whitespace collapse, and trim.  The implementation is native
// C++ so inference never imports Python or PyTorch.
std::string clean_t5_prompt(std::string_view text);

} // namespace trtmc::wan2_2
