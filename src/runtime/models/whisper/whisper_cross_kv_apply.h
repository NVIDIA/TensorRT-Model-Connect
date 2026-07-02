/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/whisper/whisper_cross_kv_plan.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace trtmc {

enum class WhisperCrossKvBufferKind {
    K,
    V,
};

struct WhisperCrossKvApplyStats {
    int32_t zero_ops{0};
    int32_t copy_ops{0};
};

template <typename ZeroFn, typename CopyFn>
inline bool apply_whisper_cross_kv_plan(const WhisperCrossKvPlan& plan, std::size_t layer_count,
                                        ZeroFn&& zero_encoder_padding, CopyFn&& copy_cross_buffer,
                                        std::string& error,
                                        WhisperCrossKvApplyStats* stats = nullptr) {
    if (plan.buffer_bytes == 0) {
        error = "invalid whisper cross-kv plan";
        return false;
    }

    if (plan.zero_pad_encoder_output) {
        if (!zero_encoder_padding(plan.valid_bytes, plan.pad_bytes)) {
            error = "failed to zero whisper encoder padding";
            return false;
        }
        if (stats != nullptr) {
            ++stats->zero_ops;
        }
    }

    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        if (!copy_cross_buffer(layer, WhisperCrossKvBufferKind::K, plan.buffer_bytes)) {
            error = "failed to copy whisper cross_k";
            return false;
        }
        if (stats != nullptr) {
            ++stats->copy_ops;
        }

        if (!copy_cross_buffer(layer, WhisperCrossKvBufferKind::V, plan.buffer_bytes)) {
            error = "failed to copy whisper cross_v";
            return false;
        }
        if (stats != nullptr) {
            ++stats->copy_ops;
        }
    }

    return true;
}

} // namespace trtmc
