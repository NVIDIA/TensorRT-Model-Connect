/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace trtmc::flux_batch {

inline std::vector<int> plan_chunks(int total, int cap) {
    if (total < 1) {
        throw std::invalid_argument("total must be >= 1");
    }
    if (cap < 1) {
        throw std::invalid_argument("cap must be >= 1");
    }
    std::vector<int> plan;
    while (total > 0) {
        const int n = std::min(total, cap);
        plan.push_back(n);
        total -= n;
    }
    return plan;
}

} // namespace trtmc::flux_batch
