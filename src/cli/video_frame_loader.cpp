/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/video_frame_loader.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::cli {
namespace {

bool is_jpeg_path(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == ".jpg" || extension == ".jpeg";
}

} // namespace

DecodedVideoClip decode_video_clip(IVideoTrackingPipeline& tracker,
                                   std::vector<std::filesystem::path> paths) {
    if (paths.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("track-hoi frame count exceeds the runtime limit");

    DecodedVideoClip clip;
    clip.paths = std::move(paths);
    clip.owned_frames.reserve(clip.paths.size());
    clip.views.reserve(clip.paths.size());

    const bool all_jpeg =
        !clip.paths.empty() && std::all_of(clip.paths.begin(), clip.paths.end(), is_jpeg_path);
    auto* batch_loader = all_jpeg ? dynamic_cast<IVideoFrameBatchLoader*>(&tracker) : nullptr;
    if (batch_loader != nullptr) {
        clip.frame_decode_mode = "model_batch";
        clip.frame_decode_max_concurrency = batch_loader->max_video_frame_load_concurrency();
        if (clip.frame_decode_max_concurrency == 0U) {
            throw std::runtime_error(
                "track-hoi model batch loader reported zero maximum concurrency");
        }
        std::vector<std::string> ordered_paths;
        ordered_paths.reserve(clip.paths.size());
        for (const auto& path : clip.paths)
            ordered_paths.push_back(path.string());
        clip.owned_frames = batch_loader->load_video_frames(ordered_paths);
        if (clip.owned_frames.size() != clip.paths.size()) {
            throw std::runtime_error(
                "track-hoi model batch loader returned an invalid frame count");
        }
    } else {
        for (const auto& path : clip.paths)
            clip.owned_frames.push_back(tracker.load_video_frame(path.string()));
    }

    for (std::size_t index = 0; index < clip.owned_frames.size(); ++index) {
        if (clip.owned_frames[index].empty()) {
            throw std::runtime_error("track-hoi failed to decode frame: " +
                                     clip.paths[index].string());
        }
    }
    for (const auto& frame : clip.owned_frames)
        clip.views.push_back(frame.view());
    return clip;
}

} // namespace trtmc::cli
