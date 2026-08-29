/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/pipeline_plugin.h"

namespace trtmc {

// Materialize, authenticate, and load the Ref2VA-owned ATen plugin DSO before
// TensorRT sees any plan that references its creators.
void load_minimax_h3_native_plugin(const PipelineContext& ctx);

} // namespace trtmc
