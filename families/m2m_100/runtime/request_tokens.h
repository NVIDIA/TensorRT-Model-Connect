/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

inline void m2m_100_apply_source_language_token(std::vector<int32_t>& ids, int32_t eos_token_id,
                                                int32_t source_language_token_id) {
    if (source_language_token_id < 0)
        return;

    if (ids.size() >= 2 && ids[ids.size() - 2] == eos_token_id) {
        ids.back() = source_language_token_id;
        return;
    }
    if (ids.empty() || ids.back() != eos_token_id)
        ids.push_back(eos_token_id);
    ids.push_back(source_language_token_id);
}

inline int32_t m2m_100_apply_forced_bos_token(int32_t selected_token_id, int32_t decoder_step,
                                              int32_t forced_bos_token_id) {
    return decoder_step == 0 && forced_bos_token_id >= 0 ? forced_bos_token_id : selected_token_id;
}

} // namespace trtmc
