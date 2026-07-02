/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace trtmc {

// Opaque base for per-step state during autoregressive generation.
// Recurrent and hybrid models own their concrete step state in family-local
// runtime code.
class IStepState {
  public:
    virtual ~IStepState() = default;
};

} // namespace trtmc
