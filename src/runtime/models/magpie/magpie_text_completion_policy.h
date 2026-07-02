/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace trtmc {

inline bool scan_xattn_peak(const float* xattn, int32_t text_length, int32_t threshold,
                            int32_t& max_peak_pos) {
    if (xattn == nullptr || text_length <= 0) {
        return false;
    }

    int32_t peak_pos = 0;
    float peak_val = xattn[0];
    for (int32_t p = 1; p < text_length; ++p) {
        if (xattn[p] > peak_val) {
            peak_val = xattn[p];
            peak_pos = p;
        }
    }
    if (peak_pos > max_peak_pos) {
        max_peak_pos = peak_pos;
    }
    return max_peak_pos >= threshold;
}

inline bool update_magpie_text_consumed_from_cross_attn(const float* xattn, int32_t text_length,
                                                        int32_t threshold, int32_t& max_peak_pos,
                                                        bool& text_consumed) {
    if (text_consumed) {
        return false;
    }
    if (!scan_xattn_peak(xattn, text_length, threshold, max_peak_pos)) {
        return false;
    }
    text_consumed = true;
    return true;
}

inline bool update_magpie_text_consumed_from_heuristic(int32_t estimated_frames, int32_t frame,
                                                       bool& text_consumed) {
    if (text_consumed || estimated_frames <= 0 || frame < estimated_frames) {
        return false;
    }
    text_consumed = true;
    return true;
}

inline bool advance_magpie_finished_limit(bool text_consumed, int32_t finished_limit,
                                          int32_t& frames_past_text_consumed) {
    if (!text_consumed) {
        return false;
    }
    ++frames_past_text_consumed;
    return finished_limit > 0 && frames_past_text_consumed >= finished_limit;
}

} // namespace trtmc
