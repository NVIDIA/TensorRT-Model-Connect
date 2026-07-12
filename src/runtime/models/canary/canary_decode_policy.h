/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

struct CanaryDecodeLoopResult {
    std::vector<int32_t> output_ids;
    bool prefill_failed{false};
    bool decode_failed{false};
    std::string error;
};

template <typename StepFn, typename SelectFn>
inline CanaryDecodeLoopResult
run_canary_decode_loop(const std::vector<int32_t>& initial_tokens, int32_t max_new_tokens,
                       int32_t eot_token_id, StepFn&& run_step, SelectFn&& select_next_token) {
    CanaryDecodeLoopResult result;
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

struct CanaryBeamHypothesis {
    std::vector<int32_t> output_ids;
    double score{0.0};
    bool finished{false};
};

inline double canary_log_normalizer(const std::vector<float>& logits) {
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    double exp_sum = 0.0;
    for (const float logit : logits) {
        exp_sum += std::exp(static_cast<double>(logit - max_logit));
    }
    return static_cast<double>(max_logit) + std::log(exp_sum);
}

inline std::vector<int32_t> canary_top_token_indices(const std::vector<float>& logits,
                                                     int32_t beam_size) {
    std::vector<int32_t> indices(logits.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int32_t>(i);
    }
    const int32_t top_count = std::min<int32_t>(beam_size, static_cast<int32_t>(indices.size()));
    std::partial_sort(indices.begin(), indices.begin() + top_count, indices.end(),
                      [&logits](int32_t lhs, int32_t rhs) {
                          const float lhs_logit = logits[static_cast<std::size_t>(lhs)];
                          const float rhs_logit = logits[static_cast<std::size_t>(rhs)];
                          return lhs_logit == rhs_logit ? lhs < rhs : lhs_logit > rhs_logit;
                      });
    indices.resize(static_cast<std::size_t>(top_count));
    return indices;
}

inline void append_canary_beam_candidates(const CanaryBeamHypothesis& beam,
                                          const std::vector<float>& logits, int32_t eot_token_id,
                                          int32_t beam_size,
                                          std::vector<CanaryBeamHypothesis>& candidates) {
    const double log_normalizer = canary_log_normalizer(logits);
    for (const int32_t token : canary_top_token_indices(logits, beam_size)) {
        CanaryBeamHypothesis candidate = beam;
        candidate.output_ids.push_back(token);
        candidate.score +=
            static_cast<double>(logits[static_cast<std::size_t>(token)]) - log_normalizer;
        candidate.finished = token == eot_token_id;
        candidates.push_back(std::move(candidate));
    }
}

template <typename LogitsFn>
inline bool
expand_canary_beam(const CanaryBeamHypothesis& beam, const std::vector<int32_t>& initial_tokens,
                   int32_t eot_token_id, int32_t beam_size, LogitsFn& logits_for_prefix,
                   std::vector<CanaryBeamHypothesis>& candidates, CanaryDecodeLoopResult& result) {
    std::vector<int32_t> prefix = initial_tokens;
    prefix.insert(prefix.end(), beam.output_ids.begin(), beam.output_ids.end());
    std::vector<float> logits;
    if (!logits_for_prefix(prefix, logits, result.error)) {
        result.decode_failed = true;
        return false;
    }
    if (logits.empty()) {
        result.decode_failed = true;
        result.error = "Canary beam search received empty logits";
        return false;
    }
    append_canary_beam_candidates(beam, logits, eot_token_id, beam_size, candidates);
    return true;
}

template <typename LogitsFn>
inline bool collect_canary_beam_candidates(const std::vector<CanaryBeamHypothesis>& beams,
                                           const std::vector<int32_t>& initial_tokens,
                                           int32_t eot_token_id, int32_t beam_size,
                                           LogitsFn& logits_for_prefix,
                                           std::vector<CanaryBeamHypothesis>& candidates,
                                           bool& all_finished, CanaryDecodeLoopResult& result) {
    all_finished = true;
    for (const auto& beam : beams) {
        if (beam.finished) {
            candidates.push_back(beam);
            continue;
        }
        all_finished = false;
        if (!expand_canary_beam(beam, initial_tokens, eot_token_id, beam_size, logits_for_prefix,
                                candidates, result)) {
            return false;
        }
    }
    return true;
}

inline void rank_canary_beam_candidates(std::vector<CanaryBeamHypothesis>& candidates,
                                        int32_t beam_size) {
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const CanaryBeamHypothesis& lhs, const CanaryBeamHypothesis& rhs) {
                         if (lhs.score != rhs.score) {
                             return lhs.score > rhs.score;
                         }
                         return lhs.output_ids < rhs.output_ids;
                     });
    if (candidates.size() > static_cast<std::size_t>(beam_size)) {
        candidates.resize(static_cast<std::size_t>(beam_size));
    }
}

template <typename LogitsFn>
inline CanaryDecodeLoopResult
run_canary_beam_search(const std::vector<int32_t>& initial_tokens, int32_t max_new_tokens,
                       int32_t eot_token_id, int32_t beam_size, LogitsFn&& logits_for_prefix) {
    CanaryDecodeLoopResult result;
    if (max_new_tokens <= 0 || beam_size <= 0)
        return result;

    std::vector<CanaryBeamHypothesis> beams(1);
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        std::vector<CanaryBeamHypothesis> candidates;
        bool all_finished = false;
        if (!collect_canary_beam_candidates(beams, initial_tokens, eot_token_id, beam_size,
                                            logits_for_prefix, candidates, all_finished, result)) {
            return result;
        }
        if (all_finished) {
            break;
        }

        rank_canary_beam_candidates(candidates, beam_size);
        beams = std::move(candidates);
    }

    if (!beams.empty()) {
        result.output_ids = std::move(beams.front().output_ids);
    }
    return result;
}

} // namespace trtmc
