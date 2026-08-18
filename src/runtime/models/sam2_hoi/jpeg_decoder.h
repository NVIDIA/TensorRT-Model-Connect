/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace trtmc::sam2_hoi {

// Decode JPEG with the libjpeg RGB path used by Pillow. An empty frame denotes
// file or decode failure, matching the generic image-loader contract.
VideoFrame decode_jpeg_pillow_rgb(const std::string& path);

inline constexpr std::size_t kMaxConcurrentJpegDecodes = 5U;

// Decode an ordered JPEG clip with bounded concurrency. Every scheduled decode
// completes before this function returns or rethrows the lowest-index failure.
std::vector<VideoFrame>
decode_jpeg_paths_bounded(const std::vector<std::string>& paths, std::size_t max_concurrency,
                          const std::function<VideoFrame(const std::string&)>& decoder);

std::vector<VideoFrame> decode_jpeg_pillow_rgb_batch(const std::vector<std::string>& paths);

} // namespace trtmc::sam2_hoi
