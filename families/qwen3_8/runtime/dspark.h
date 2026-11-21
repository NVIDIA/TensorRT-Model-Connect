/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "families/qwen3_8/runtime/external_draft.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {
struct Qwen38DSparkConfig {
    int32_t hidden, vocab, capacity, feature_count, rows, mask_token;
    bool sample_from_anchor, causal, confidence;
};
using Qwen38DSparkProposal = Qwen38DraftProposal;

// Owns only draft state. The caller observes confirmed target rows; rejected
// proposals are never executed on the recurrent target and need no rollback.
class Qwen38DSpark final : public Qwen38ExternalDraft {
  public:
    Qwen38DSpark(Qwen38DSparkConfig config, std::unique_ptr<ITrtModule> draft,
                 std::unique_ptr<ITrtModule> heads, std::vector<char> embeddings);
    void reset(ITrtModule& target, std::size_t prompt_length, int32_t max_new_tokens) override;
    void observe(ITrtModule& target) override;
    std::vector<Qwen38DSparkProposal> propose(int32_t anchor, const Qwen38SamplingParams& sampling,
                                              const std::vector<int32_t>& history, int32_t budget,
                                              uint64_t& rng) override;
    const char* name() const override { return "DSpark"; }

  private:
    Qwen38DSparkConfig config_;
    std::unique_ptr<ITrtModule> draft_, heads_;
    std::vector<char> embeddings_;
    std::vector<uint16_t> features_;
    std::size_t feature_width_;
    int32_t seen_{0};
};
} // namespace trtmc
