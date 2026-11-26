/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

namespace trtmc::muse_glimmer {
struct TreeNode {
    int32_t token;
    int32_t parent;
};
using VerifyTree = std::vector<TreeNode>;
// Score the 16 candidates from one proposal row given this branch's predecessor.
using SelectRow = std::function<std::vector<float>(int32_t row, int32_t predecessor)>;
VerifyTree dflash2_tree(int32_t anchor, const std::vector<int32_t>& ids, const SelectRow& select);
} // namespace trtmc::muse_glimmer
