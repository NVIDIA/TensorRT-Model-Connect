/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "native_proposal.h"
#include "sampler.h"

namespace trtmc::nemotron_bolt {
class NativeLiLiCorr {
  public:
    NativeLiLiCorr(IBackend& engine, const NativeLanguage::PlanReader& read);
    void reset(int first_position);
    void accepted(const std::vector<uint16_t>& target_features);
    std::array<BoltDraftChoice, 7> propose(int root, const std::vector<char>& embeddings,
                                           BoltVerificationSampler& sampler);

  private:
    NativeProposal proposal_;
    std::unique_ptr<ITrtModule> project_, vocab_, lattice_;
    std::vector<float> anchor_;
};
} // namespace trtmc::nemotron_bolt
