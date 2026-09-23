/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/llama/runtime/speculative/engine.h"
#include "trtmc/runtime/family_factory.h"

namespace trtmc::llama::speculative {

struct Acceptance {
    int length;
    bool stopped;
    std::vector<std::int32_t> emitted;
};

// EAGLE3 policy, separate from the model/attention Engine ABI.
// All GPU computation is compiled TensorRT graph computation.
class Eagle3 {
  public:
    Eagle3(const FamilyContext& context, Engine& target, Engine& draft, int depth);
    ~Eagle3();
    void prefill(const std::vector<std::int32_t>& prompt, int root, FeatureView target_features);
    void propose_and_verify(int root, int committed, int remaining, int width, bool ignore_eos);
    Acceptance accept(int committed);
    void feedback(int length);
    void finish();

  private:
    void order(cudaStream_t consumer, cudaStream_t producer);
    ITrtModule& prepare(int width, int depth, int offset, cudaStream_t ready);
    Engine &target_, &draft_;
    int max_depth_, depth_ = 0, width_ = 1, rows_ = 1;
    std::unique_ptr<ITrtModule> mapping_, features_, kv_;
    std::vector<std::unique_ptr<ITrtModule>> prepare_[2], accept_[2];
    DeviceTensor starts_, tokens_, valid_, limits_, root_;
    std::shared_ptr<void> upload_, readback_, event_;
    DeviceStep draft_result_{}, target_result_{};
    bool root_on_device_ = false;
};

} // namespace trtmc::llama::speculative
