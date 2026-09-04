/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// State lifecycle for VoiceChat's static, single-token thinker engine.

#include "trtmc/runtime/tensor.h"

namespace trtmc {

class ITrtModule;

class VoiceChatThinkerInferenceState {
  public:
    virtual ~VoiceChatThinkerInferenceState() = default;

    virtual void reset() = 0;
    virtual void bind_to(ITrtModule& module) = 0;
    virtual void prepare_step(TensorMap& inputs) = 0;
    virtual void advance() = 0;
    virtual bool ok() const = 0;
};

} // namespace trtmc
