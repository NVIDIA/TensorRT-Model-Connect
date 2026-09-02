/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <cstdint>

namespace trtmc::cli {

struct MiniMaxH3VideoContract {
    int32_t num_frames{0};
    int32_t height{0};
    int32_t width{0};
};

// Resolve the exact public H3 output profile before invoking the model plugin.
// The resolved canvas is written into the request so the CLI validator and the
// plugin consume the same dimensions even when FL2VA derives aspect ratio from
// a keyframe.
MiniMaxH3VideoContract resolve_minimax_h3_video_contract(VideoGenerationRequest& request);

} // namespace trtmc::cli
