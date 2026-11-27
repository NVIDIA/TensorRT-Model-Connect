/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/qwen3_8/runtime/external_draft.h"

#include <functional>

namespace trtmc {
// Score the block-8 checkpoint's 16 candidates at a depth given one actual predecessor.
// TensorRT's selector input binding borrows the candidate storage during this call.
using Qwen38DFlash2Select = std::function<std::vector<float>(int32_t depth, int32_t predecessor,
                                                             std::vector<int32_t>& candidates)>;
Qwen38VerifyTree qwen38_dflash2_tree(int32_t anchor, const std::vector<int32_t>& ids,
                                     const Qwen38DFlash2Select& select);
} // namespace trtmc
