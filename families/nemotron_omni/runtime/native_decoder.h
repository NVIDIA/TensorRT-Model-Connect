/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc::nemotron_omni {

class NativeDecoder {
  public:
    explicit NativeDecoder(const FamilyContext& context);
    void reset();
    std::vector<float> step(const std::vector<float>& embedding);
    int position() const noexcept { return position_; }
    int capacity() const noexcept { return capacity_; }
    int hidden() const noexcept { return hidden_; }
    int vocab() const noexcept { return vocab_; }

  private:
    struct State {
        std::string output;
        DeviceTensor storage;
        bool append;
    };
    // Module/context must be destroyed before externally bound allocations.
    std::vector<State> states_;
    std::unique_ptr<ITrtModule> module_;
    int hidden_{0}, vocab_{0}, capacity_{0}, position_{0};
    DType dtype_{DType::kFloat32};
};

} // namespace trtmc::nemotron_omni
