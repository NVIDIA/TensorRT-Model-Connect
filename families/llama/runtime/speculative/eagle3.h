/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/llama/runtime/speculative/engine.h"

namespace trtmc::llama::speculative {

struct CandidateTree {
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> parents;
};

// Method-specific draft conditioning and proposal policy. The engine adapter
// and its compiler/runtime tensor contract know nothing about EAGLE3.
class Eagle3 {
  public:
    Eagle3(Engine& draft, const std::vector<std::int32_t>& mapping, int depth, int width);
    void prefill(const std::vector<std::int32_t>& prompt, int root, FeatureView target_features);
    CandidateTree propose(int root, int committed, int remaining);
    void feedback(const CandidateTree& tree, const std::vector<std::int32_t>& path,
                  FeatureView target_features, int bonus, int committed);
    FeatureView features() const { return result_.features; }

  private:
    Engine& draft_;
    const std::vector<std::int32_t>& mapping_;
    int depth_, width_;
    StepResult result_;
};

} // namespace trtmc::llama::speculative
