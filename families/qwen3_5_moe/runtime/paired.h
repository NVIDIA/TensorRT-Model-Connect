/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/qwen3_5_moe/runtime/sampler.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

// Draft state is reconstructed from accepted target features each round. The
// target verifies sequentially, so rejected branches never mutate DeltaNet state.
struct Qwen35MoePairedConfig {
    std::string variant;
    int32_t hidden_size, vocab_size, capacity, max_input_length;
    std::vector<int32_t> feature_layers;
    int32_t rows, mask_token, top_k, verify_size;
};

struct Qwen35MoeProposal {
    int32_t token, parent, depth;
    std::vector<float> probabilities; // Chain sampling q; tree nodes need no q.
};

class Qwen35MoePairedDraft {
  public:
    Qwen35MoePairedDraft(Qwen35MoePairedConfig config, std::unique_ptr<ITrtModule> draft);
    void reset(ITrtModule& target, std::size_t prompt_length, int32_t max_new_tokens,
               const Qwen35MoeSamplingParams& sampling);
    void observe(ITrtModule& target, int32_t position);
    std::vector<Qwen35MoeProposal> propose(int32_t anchor, const Qwen35MoeSamplingParams& sampling,
                                           int32_t budget, std::uint64_t& rng);
    bool tree() const { return config_.variant == "dflash_ddtree"; }

  private:
    Qwen35MoePairedConfig config_;
    std::unique_ptr<ITrtModule> draft_;
    std::size_t feature_width_;
    std::vector<std::uint16_t> features_;
    int32_t seen_{0};
};

std::unique_ptr<Qwen35MoePairedDraft>
create_qwen35_paired(const FamilyContext& context, int32_t hidden, int32_t layers, int32_t vocab,
                     int32_t capacity, const std::string& precision, int32_t tensor_parallel_size,
                     bool has_mtp);

} // namespace trtmc
