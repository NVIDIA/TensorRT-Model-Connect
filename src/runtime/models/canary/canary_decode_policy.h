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
    int32_t state_slot{0};
    std::vector<float> logits;
};

struct CanaryBeamCandidate {
    CanaryBeamHypothesis hypothesis;
    int32_t parent_slot{-1};
    int32_t token{-1};
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

inline double canary_normalized_beam_score(const CanaryBeamHypothesis& hypothesis,
                                           float length_penalty) {
    if (length_penalty <= 0.0F || hypothesis.output_ids.empty()) {
        return hypothesis.score;
    }
    const double length = static_cast<double>(hypothesis.output_ids.size());
    return hypothesis.score / std::pow(length, static_cast<double>(length_penalty));
}

inline void append_canary_beam_candidates(const CanaryBeamHypothesis& beam, int32_t eot_token_id,
                                          int32_t beam_size,
                                          std::vector<CanaryBeamCandidate>& candidates) {
    const double log_normalizer = canary_log_normalizer(beam.logits);
    for (const int32_t token : canary_top_token_indices(beam.logits, beam_size)) {
        CanaryBeamHypothesis candidate = beam;
        candidate.output_ids.push_back(token);
        candidate.score +=
            static_cast<double>(beam.logits[static_cast<std::size_t>(token)]) - log_normalizer;
        candidate.finished = token == eot_token_id;
        candidate.logits.clear();
        candidates.push_back({std::move(candidate), beam.state_slot, token});
    }
}

inline bool collect_canary_beam_candidates(const std::vector<CanaryBeamHypothesis>& beams,
                                           int32_t eot_token_id, int32_t beam_size,
                                           std::vector<CanaryBeamCandidate>& candidates,
                                           bool& all_finished, CanaryDecodeLoopResult& result) {
    all_finished = true;
    for (const auto& beam : beams) {
        if (beam.finished) {
            candidates.push_back({beam, -1, -1});
            continue;
        }
        all_finished = false;
        if (beam.logits.empty()) {
            result.decode_failed = true;
            result.error = "Canary beam search received empty logits";
            return false;
        }
        append_canary_beam_candidates(beam, eot_token_id, beam_size, candidates);
    }
    return true;
}

inline void rank_canary_beam_candidates(std::vector<CanaryBeamCandidate>& candidates,
                                        int32_t beam_size, float length_penalty) {
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [length_penalty](const CanaryBeamCandidate& lhs, const CanaryBeamCandidate& rhs) {
            const double lhs_score = canary_normalized_beam_score(lhs.hypothesis, length_penalty);
            const double rhs_score = canary_normalized_beam_score(rhs.hypothesis, length_penalty);
            if (lhs_score != rhs_score) {
                return lhs_score > rhs_score;
            }
            return lhs.hypothesis.output_ids < rhs.hypothesis.output_ids;
        });
    if (candidates.size() > static_cast<std::size_t>(beam_size)) {
        candidates.resize(static_cast<std::size_t>(beam_size));
    }
}

template <typename AdvanceFn>
inline bool advance_canary_beam_candidates(int32_t generation, int32_t max_new_tokens,
                                           std::vector<CanaryBeamCandidate>& candidates,
                                           AdvanceFn& advance,
                                           std::vector<CanaryBeamHypothesis>& next_beams,
                                           CanaryDecodeLoopResult& result) {
    const bool final_step = generation + 1 == max_new_tokens;
    next_beams.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        auto& candidate = candidates[i];
        if (!candidate.hypothesis.finished && !final_step) {
            const int32_t child_slot = static_cast<int32_t>(i);
            if (!advance(generation, candidate.parent_slot, child_slot, candidate.token,
                         candidate.hypothesis.logits, result.error)) {
                result.decode_failed = true;
                return false;
            }
            if (candidate.hypothesis.logits.empty()) {
                result.decode_failed = true;
                result.error = "Canary beam search received empty logits after advance";
                return false;
            }
            candidate.hypothesis.state_slot = child_slot;
        } else {
            candidate.hypothesis.state_slot = -1;
        }
        next_beams.push_back(std::move(candidate.hypothesis));
    }
    return true;
}

template <typename PrefillFn, typename AdvanceFn>
inline CanaryDecodeLoopResult run_canary_beam_search(const std::vector<int32_t>& initial_tokens,
                                                     int32_t max_new_tokens, int32_t eot_token_id,
                                                     int32_t beam_size, float length_penalty,
                                                     PrefillFn&& prefill, AdvanceFn&& advance) {
    CanaryDecodeLoopResult result;
    if (max_new_tokens <= 0 || beam_size <= 0) {
        return result;
    }

    std::vector<CanaryBeamHypothesis> beams(1);
    if (!prefill(initial_tokens, beams.front().logits, result.error)) {
        result.prefill_failed = true;
        return result;
    }
    if (beams.front().logits.empty()) {
        result.decode_failed = true;
        result.error = "Canary beam search received empty prefill logits";
        return result;
    }

    for (int32_t step = 0; step < max_new_tokens; ++step) {
        std::vector<CanaryBeamCandidate> candidates;
        bool all_finished = false;
        if (!collect_canary_beam_candidates(beams, eot_token_id, beam_size, candidates,
                                            all_finished, result)) {
            return result;
        }
        if (all_finished) {
            break;
        }

        rank_canary_beam_candidates(candidates, beam_size, length_penalty);
        std::vector<CanaryBeamHypothesis> next_beams;
        if (!advance_canary_beam_candidates(step, max_new_tokens, candidates, advance, next_beams,
                                            result)) {
            return result;
        }
        beams = std::move(next_beams);
    }

    if (!beams.empty()) {
        result.output_ids = std::move(beams.front().output_ids);
    }
    return result;
}

} // namespace trtmc
