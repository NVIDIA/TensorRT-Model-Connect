/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace trtmc {

struct WhisperCrossKvPlan {
    std::size_t buffer_bytes{0};
    std::size_t valid_bytes{0};
    std::size_t pad_bytes{0};
    bool zero_pad_encoder_output{false};
};

inline WhisperCrossKvPlan make_whisper_cross_kv_plan(int32_t encoder_seq_length,
                                                     int32_t hidden_size,
                                                     int32_t actual_encoder_seq_length) {
    WhisperCrossKvPlan plan;
    if (encoder_seq_length <= 0 || hidden_size <= 0) {
        return plan;
    }

    const std::size_t encoder_tokens = static_cast<std::size_t>(encoder_seq_length);
    const std::size_t hidden = static_cast<std::size_t>(hidden_size);
    plan.buffer_bytes = encoder_tokens * hidden * sizeof(float);

    if (actual_encoder_seq_length <= 0 || actual_encoder_seq_length >= encoder_seq_length) {
        plan.valid_bytes = plan.buffer_bytes;
        return plan;
    }

    plan.valid_bytes = static_cast<std::size_t>(actual_encoder_seq_length) * hidden * sizeof(float);
    plan.pad_bytes = plan.buffer_bytes - plan.valid_bytes;
    plan.zero_pad_encoder_output = plan.pad_bytes > 0;
    return plan;
}

} // namespace trtmc
