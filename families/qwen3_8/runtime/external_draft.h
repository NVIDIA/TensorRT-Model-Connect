/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/qwen3_8/runtime/sampler.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace trtmc {
struct Qwen38DraftProposal {
    int32_t token;
    std::vector<float> probabilities;
};
// The verifier follows target samples through the retained proposal tree. A chain
// carries proposal probabilities instead and uses rejection sampling.
struct Qwen38VerifyNode {
    int32_t token;
    int32_t parent;
};
using Qwen38VerifyTree = std::vector<Qwen38VerifyNode>;
using Qwen38DraftRound = std::variant<std::vector<Qwen38DraftProposal>, Qwen38VerifyTree>;
// Qwen-owned external draft contract. Target verification and confirmed-state
// observation are identical; each companion owns its own proposal computation.
class Qwen38ExternalDraft {
  public:
    virtual ~Qwen38ExternalDraft() = default;
    virtual void validate_sampling(const Qwen38SamplingParams&) const {}
    virtual void reset(ITrtModule& target, std::size_t prompt_length, int32_t max_new_tokens) = 0;
    virtual void observe(ITrtModule& target) = 0;
    virtual Qwen38DraftRound propose(int32_t anchor, const Qwen38SamplingParams& sampling,
                                     const std::vector<int32_t>& history, int32_t budget,
                                     uint64_t& rng) = 0;
    virtual const char* name() const = 0;
};
} // namespace trtmc
