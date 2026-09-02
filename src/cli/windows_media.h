/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <string>
#include <string_view>

namespace trtmc::cli {

bool is_mp4_path(std::string_view path);

// Write synchronized H.264 video and optional AAC audio using the Windows
// Media Foundation codecs shipped with the operating system. No FFmpeg or
// other runtime media dependency is involved.
void write_mp4(const VideoResult& result, const std::string& path);

// Decode a Windows-supported media container into the public THWC RGB video
// and interleaved float-audio value type used by Ref2VA.
VideoClipInput read_video_file(const std::string& path);

} // namespace trtmc::cli
