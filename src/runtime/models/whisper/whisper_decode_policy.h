#pragma once

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

template <typename StepFn, typename SelectFn>
inline WhisperDecodeLoopResult
run_whisper_decode_loop(const std::vector<int32_t>& initial_tokens, int32_t max_new_tokens,
                        int32_t eot_token_id, StepFn&& run_step, SelectFn&& select_next_token) {
    WhisperDecodeLoopResult result;
    std::vector<float> logits;

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

} // namespace trtmc
