/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/image.h"

#include <vector>

namespace trtmc::muse_glimmer {
// Convert validated byte or normalized float HWC input to contiguous RGB bytes.
std::vector<std::uint8_t> visual_rgb(const internal::ImageView& image);

// Raster-order patches, each laid out as two frames x RGB x patch height x patch width.
// A null second frame duplicates the first for a still image or an odd final video group.
std::vector<float> visual_patches(const internal::ImageView& first,
                                  const internal::ImageView* second, int rows, int cols,
                                  int patch_size);
} // namespace trtmc::muse_glimmer
