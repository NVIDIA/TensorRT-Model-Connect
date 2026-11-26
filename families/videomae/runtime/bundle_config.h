/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/videomae/runtime/clip.h"
#include "trtmc/runtime/trt_module.h"

#include <string_view>

namespace trtmc::videomae {

struct RuntimeConfig {
    ClipGeometry geometry;
    PixelNormalization normalization;
};

// Parses the bundle's runtime.json: one object with exactly the keys the
// builder writes, integers as JSON integers, three finite statistics per
// channel, no repeated key. Throws std::runtime_error otherwise.
RuntimeConfig parse_runtime_config(std::string_view text);

// Checks the engine's tensors against the clip geometry before first use:
// pixel_values float32 [1, frames, 3, size, size]; visible_index and
// masked_index int32 of rank 1; logits float32 [1, M, tube values].
void validate_engine(const ITrtModule& engine, const ClipGeometry& geometry);

} // namespace trtmc::videomae
