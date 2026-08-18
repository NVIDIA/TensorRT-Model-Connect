/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace trtmc::cli {

struct DecodedVideoClip {
    std::vector<std::filesystem::path> paths;
    std::vector<VideoFrame> owned_frames;
    std::vector<VideoFrameView> views;
    std::string frame_decode_mode{"serial"};
    std::size_t frame_decode_max_concurrency{1U};
};

DecodedVideoClip decode_video_clip(IVideoTrackingPipeline& tracker,
                                   std::vector<std::filesystem::path> paths);

} // namespace trtmc::cli
