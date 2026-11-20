/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

struct WhisperDecodeLoopResult {
    std::vector<int32_t> output_ids;
    bool prefill_failed{false};
    bool decode_failed{false};
    std::string error;
};

// One step's logits where the step left them, for a step that does not copy
// them into a vector.
struct WhisperLogitsView {
    const float* data{nullptr};
    std::size_t size{0};

    bool empty() const { return size == 0; }
};

// The decode loop over the step's own holder of its logits: run_step fills a
// Logits, select_next_token reads it.
template <typename Logits, typename StepFn, typename SelectFn>
inline WhisperDecodeLoopResult
run_whisper_decode_loop_over(const std::vector<int32_t>& initial_tokens, int32_t max_new_tokens,
                             int32_t eot_token_id, StepFn&& run_step,
                             SelectFn&& select_next_token) {
    WhisperDecodeLoopResult result;
    Logits logits{};

    for (const int32_t token : initial_tokens) {
        if (!run_step(token, logits, result.error)) {
            result.prefill_failed = true;
            return result;
        }
    }

    if (max_new_tokens <= 0 || logits.empty()) {
        return result;
    }

    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const int32_t next_token = select_next_token(logits);
        result.output_ids.push_back(next_token);

        if (next_token == eot_token_id) {
            break;
        }

        if (!run_step(next_token, logits, result.error)) {
            result.decode_failed = true;
            break;
        }
    }

    return result;
}

template <typename StepFn, typename SelectFn>
inline WhisperDecodeLoopResult
run_whisper_decode_loop(const std::vector<int32_t>& initial_tokens, int32_t max_new_tokens,
                        int32_t eot_token_id, StepFn&& run_step, SelectFn&& select_next_token) {
    return run_whisper_decode_loop_over<std::vector<float>>(
        initial_tokens, max_new_tokens, eot_token_id, std::forward<StepFn>(run_step),
        std::forward<SelectFn>(select_next_token));
}

} // namespace trtmc
