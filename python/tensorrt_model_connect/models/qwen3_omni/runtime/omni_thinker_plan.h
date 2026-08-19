/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace trtmc {

inline bool omni_thinker_should_stop(int32_t token_id, int32_t eos_token_id) {
    return eos_token_id >= 0 && token_id == eos_token_id;
}

} // namespace trtmc
