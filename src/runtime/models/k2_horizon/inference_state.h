/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/tensor.h"

#include <cstddef>
#include <cstdint>

namespace trtmc {

class ITrtModule;
using TrtModule = ITrtModule;

// Family-owned lifecycle contract for the qualified fixed native KV state.
class K2HorizonInferenceState {
  public:
    virtual ~K2HorizonInferenceState() = default;

    virtual void reset() = 0;
    virtual void bind_to(TrtModule& module) = 0;
    virtual void prepare_step(TensorMap& inputs, int32_t seq_len = 1) = 0;
    virtual void advance(int32_t n_tokens = 1) = 0;
    virtual int32_t position() const = 0;
    virtual int32_t max_length() const = 0;
    virtual std::size_t device_memory_bytes() const = 0;
    virtual bool ok() const = 0;
};

} // namespace trtmc
