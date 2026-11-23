/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/qwen3_8/runtime/external_draft.h"

#include <memory>

namespace trtmc {
struct Qwen38DominoConfig {
    int32_t hidden, vocab, capacity, feature_count, mask_token, pool, rank, state, top_k;
};
class Qwen38Domino final : public Qwen38ExternalDraft {
  public:
    Qwen38Domino(Qwen38DominoConfig config, std::unique_ptr<ITrtModule> draft,
                 std::unique_ptr<ITrtModule> correction, std::vector<char> embeddings);
    void validate_sampling(const Qwen38SamplingParams& sampling) const override;
    void reset(ITrtModule& target, std::size_t prompt_length, int32_t max_new_tokens) override;
    void observe(ITrtModule& target) override;
    Qwen38DraftRound propose(int32_t anchor, const Qwen38SamplingParams& sampling,
                             const std::vector<int32_t>& history, int32_t budget,
                             uint64_t& rng) override;
    const char* name() const override { return "Domino"; }

  private:
    Qwen38DominoConfig config_;
    std::unique_ptr<ITrtModule> draft_, correction_;
    std::vector<char> embeddings_;
    std::vector<uint16_t> features_;
    std::size_t feature_width_;
    int32_t seen_{0};
};
} // namespace trtmc
