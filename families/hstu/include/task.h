/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "trtmc/runtime/tensor.h"
#include "trtmc/structured_ranking.h"

namespace trtmc::hstu {
// Family-owned semantics use the shared host tensor transport contract.
using FeatureTensor = trtmc::FeatureTensor;

struct TimingProfile {
    double prepare_wall_ms{};
    std::vector<double> round_wall_ms;
    std::vector<double> round_gpu_ms;
};

class IStructuredRanking : public trtmc::IStructuredRanking {
  public:
    static constexpr const char* kTask = "structured_ranking";
    const char* task() const noexcept override { return kTask; }
    virtual std::vector<FeatureTensor> score(const std::vector<FeatureTensor>& inputs) = 0;
    // Optional synchronized diagnostic profile. Each round includes GPU completion.
    virtual void set_profiling(bool) {}
    virtual TimingProfile last_profile() const { return {}; }
};

} // namespace trtmc::hstu
