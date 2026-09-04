/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace trtmc {

inline int64_t resolve_magpie_seed(int64_t session_seed, int32_t request_seed) {
    return request_seed >= 0 ? static_cast<int64_t>(request_seed) : session_seed;
}

} // namespace trtmc
