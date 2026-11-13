/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/qwen3_8/runtime/external_draft.h"

#include <memory>

namespace trtmc {
struct Qwen38DFlash2Config {
    int32_t hidden, vocab, capacity, feature_count, mask_token, window, rank, top_k;
};
// DFlash2 proposals use temperature on selector top-16, not target top-p/min-p
// or repetition filters. Target verification retains all requested filters.
std::vector<float> qwen38_dflash2_probabilities(const std::vector<float>& scores,
                                                const std::vector<int32_t>& ids, int32_t vocab,
                                                const Qwen38SamplingParams& sampling);
class Qwen38DFlash2 final : public Qwen38ExternalDraft {
  public:
    Qwen38DFlash2(Qwen38DFlash2Config config, std::unique_ptr<ITrtModule> draft,
                  std::unique_ptr<ITrtModule> selector, std::vector<char> embeddings);
    void reset(ITrtModule& target, std::size_t prompt_length, int32_t max_new_tokens) override;
    void observe(ITrtModule& target) override;
    std::vector<Qwen38DraftProposal> propose(int32_t anchor, const Qwen38SamplingParams& sampling,
                                             const std::vector<int32_t>& history, int32_t budget,
                                             uint64_t& rng) override;
    const char* name() const override { return "DFlash2"; }

  private:
    Qwen38DFlash2Config config_;
    std::unique_ptr<ITrtModule> draft_, selector_;
    std::vector<char> embeddings_;
    std::vector<uint16_t> features_;
    std::size_t feature_width_;
    int32_t seen_{0};
};
} // namespace trtmc
