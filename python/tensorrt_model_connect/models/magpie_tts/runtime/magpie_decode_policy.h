/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

namespace trtmc {

constexpr int32_t kMagpieBosToken = 2016;
constexpr int32_t kMagpieEosToken = 2017;
constexpr int32_t kMagpieAudioRange = 2016; // tokens 0..2015 are valid audio
constexpr int32_t kMagpieMinFrames = 4;
// Special tokens 2016-2023: BOS=2016, EOS=2017, CONTEXT_BOS=2018, CONTEXT_EOS=2019,
// MASK=2020, RESERVED=2021-2023. All forbidden except EOS (conditionally).
constexpr int32_t kMagpieSpecialStart = 2016;
constexpr int32_t kMagpieSpecialEnd = 2024; // exclusive

// Mask forbidden special tokens in-place: set logit to -inf for all special tokens
// except EOS (2017). If forbid_eos is true, also mask EOS.
inline void mask_magpie_special_tokens(float* logits, int32_t cb_size, bool forbid_eos) {
    for (int32_t t = kMagpieSpecialStart; t < std::min(cb_size, kMagpieSpecialEnd); ++t) {
        if (t == kMagpieEosToken && !forbid_eos)
            continue;
        logits[t] = -1e9F;
    }
}

struct FrameDecodeResult {
    std::vector<int32_t> frame_codes;
    bool eos{false};
};

inline int32_t magpie_argmax_index(const float* values, int32_t count) {
    int32_t best_id = 0;
    float best_val = values[0];
    for (int32_t i = 1; i < count; ++i) {
        if (values[i] > best_val) {
            best_val = values[i];
            best_id = i;
        }
    }
    return best_id;
}

inline FrameDecodeResult decode_magpie_frame_codes(
    const std::vector<float>& logits, int32_t num_cb, int32_t cb_size, bool greedy,
    float temperature, int32_t top_k,
    const std::function<int32_t(const float*, int32_t, float, int32_t)>& sampler,
    bool forbid_eos = false, bool force_eos = false) {
    FrameDecodeResult result;
    result.frame_codes.assign(static_cast<std::size_t>(num_cb), 0);

    // Force EOS: immediately return with eos=true (NeMo: finished_items)
    if (force_eos) {
        result.eos = true;
        return result;
    }

    for (int32_t cb = 0; cb < num_cb; ++cb) {
        const int32_t offset = cb * cb_size;
        if (offset + cb_size > static_cast<int32_t>(logits.size())) {
            continue;
        }

        // Copy logits so we can mask in-place
        std::vector<float> cb_logits_buf(logits.data() + offset, logits.data() + offset + cb_size);
        float* cb_logits = cb_logits_buf.data();

        // Mask forbidden special tokens (NeMo: clear_forbidden_logits)
        mask_magpie_special_tokens(cb_logits, cb_size, forbid_eos);

        // EOS detection via argmax (NeMo: argmax_or_multinomial_any)
        if (magpie_argmax_index(cb_logits, cb_size) == kMagpieEosToken) {
            result.eos = true;
        }

        if (greedy) {
            result.frame_codes[static_cast<std::size_t>(cb)] =
                magpie_argmax_index(cb_logits, kMagpieAudioRange);
            continue;
        }

        const int32_t sampled_id = sampler(cb_logits, cb_size, temperature, top_k);
        if (sampled_id == kMagpieEosToken) {
            result.eos = true;
            continue;
        }
        if (sampled_id >= 0 && sampled_id < kMagpieAudioRange) {
            result.frame_codes[static_cast<std::size_t>(cb)] = sampled_id;
            continue;
        }

        // Fallback: use argmax within audio range
        result.frame_codes[static_cast<std::size_t>(cb)] =
            magpie_argmax_index(cb_logits, kMagpieAudioRange);
    }

    return result;
}

inline bool should_run_magpie_periodic_check(int32_t frame, int32_t min_frames, int32_t interval) {
    return frame >= min_frames && interval > 0 && ((frame + 1) % interval == 0);
}

inline bool should_stop_magpie_on_eos(bool eos, int32_t frame, int32_t min_frames) {
    return eos && frame >= min_frames;
}

} // namespace trtmc
