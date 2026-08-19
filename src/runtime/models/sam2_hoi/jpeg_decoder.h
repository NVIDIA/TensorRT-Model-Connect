/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam2_hoi/pipeline.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace trtmc::sam2_hoi {

// Decode JPEG with the libjpeg RGB path used by Pillow. An empty frame denotes
// file or decode failure, matching the model-owned image-loader contract.
Sam2HoiVideoFrame decode_jpeg_pillow_rgb(const std::string& path);

inline constexpr std::size_t kMaxConcurrentJpegDecodes = 5U;

// Decode an ordered JPEG clip with bounded concurrency. Every scheduled decode
// completes before this function returns or rethrows the lowest-index failure.
std::vector<Sam2HoiVideoFrame>
decode_jpeg_paths_bounded(const std::vector<std::string>& paths, std::size_t max_concurrency,
                          const std::function<Sam2HoiVideoFrame(const std::string&)>& decoder);

std::vector<Sam2HoiVideoFrame> decode_jpeg_pillow_rgb_batch(const std::vector<std::string>& paths);

} // namespace trtmc::sam2_hoi
