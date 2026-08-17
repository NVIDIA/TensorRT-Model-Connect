/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::sam2 {

// Resize one float32 mask-logit plane using torch interpolate's bilinear,
// align_corners=false coordinate convention, then threshold strictly > 0.
// The reviewed reference has no optional connected-component extension, so no
// hole filling is applied.
std::vector<std::uint8_t> resizeAndThresholdMask(const float* mask_logits,
                                                 std::int32_t source_height,
                                                 std::int32_t source_width,
                                                 std::int32_t output_height,
                                                 std::int32_t output_width);

} // namespace trtmc::sam2
