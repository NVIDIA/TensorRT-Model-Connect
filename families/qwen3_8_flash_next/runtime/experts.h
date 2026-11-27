/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace trtmc::qwen3_8_flash_next {

// Gather rows by selected expert, run each needed bank exactly once for this
// chunk and restore token/rank order for the TensorRT weighted reduction.
std::vector<std::uint16_t> run_experts(ITrtModule& module, const Tensor& tokens,
                                       const Tensor& selected,
                                       const std::function<std::vector<char>(std::int32_t)>& load);

} // namespace trtmc::qwen3_8_flash_next
