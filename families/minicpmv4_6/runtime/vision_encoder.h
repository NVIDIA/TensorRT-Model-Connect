/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minicpmv4_6/runtime/image.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::minicpmv4_6 {

// Runs vision.plan once per slice; returns [slice tokens, hidden] feature rows
// at the decoder precision.
class VisionEncoder {
  public:
    VisionEncoder(std::unique_ptr<ITrtModule> module, std::int32_t hidden_size, DType dtype,
                  std::int32_t patch_width, std::int32_t max_patches);
    std::vector<std::uint8_t> encode(const VisionSlice& slice);

  private:
    std::unique_ptr<ITrtModule> module_;
    std::int32_t hidden_size_{0};
    DType dtype_{DType::kFloat16};
    std::int32_t patch_width_{0};
};

} // namespace trtmc::minicpmv4_6
