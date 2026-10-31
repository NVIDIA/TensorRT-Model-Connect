/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/reference_layout.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace trtmc::minimax_h3 {

struct ReferenceVsaVideoGrid {
    int32_t frames;
    int32_t height;
    int32_t width;
};

// Dense lengths are already token counts. Video dimensions are raw latent
// grids and receive the checkpoint's (1,2,2) patching before 128-token tiling.
using ReferenceVsaSegment = std::variant<int32_t, ReferenceVsaVideoGrid>;

struct ReferenceVsaLayout {
    int32_t source_rows{0};
    int32_t prefix_tiles{0};
    int32_t logical_tiles{0};
    std::vector<int32_t> gather_indices;
    std::vector<int32_t> untile_indices;
    std::vector<int32_t> block_sizes;
    std::vector<int32_t> regions; // Row-major [region, begin/end/keep].
};

ReferenceVsaLayout make_reference_vsa_layout(const std::vector<ReferenceVsaSegment>& segments,
                                             int32_t max_tiles = 8192);

ReferenceVsaLayout make_reference_vsa_layout(int32_t text_rows,
                                             const std::vector<ImageReferenceLatentShape>& images,
                                             ReferenceTargetLatentShape target,
                                             std::optional<ReferenceContinuationGuideShape> guide,
                                             int32_t max_tiles);

} // namespace trtmc::minimax_h3
