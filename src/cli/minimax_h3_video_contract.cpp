/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/minimax_h3_video_contract.h"

#include "runtime/models/minimax_h3/public_profile.h"

#include <stdexcept>

namespace trtmc::cli {

MiniMaxH3VideoContract resolve_minimax_h3_video_contract(VideoGenerationRequest& request) {
    MiniMaxH3VideoContract result;
    const int32_t requested_frames = request.config.video_num_frames > 0
                                         ? request.config.video_num_frames
                                         : kMiniMaxH3DefaultOutputFrames;
    result.num_frames = align_minimax_h3_num_frames(requested_frames);

    const bool fl2va = request.mode == VideoGenerationMode::kFirstLastFrameToVideoAudio;
    if (fl2va && !request.first_frame && !request.last_frame)
        throw std::invalid_argument("MiniMax-H3 FL2VA needs an endpoint keyframe");

    if (fl2va && request.config.height == 0 && request.config.width == 0) {
        // The public FL2VA contract uses the first-frame aspect when both
        // endpoints are supplied; the last frame is cover-resized/cropped.
        const VideoImageInput& anchor =
            request.first_frame ? *request.first_frame : *request.last_frame;
        const MiniMaxH3Canvas canvas = resolve_minimax_h3_canvas(anchor.width, anchor.height);
        request.config.height = canvas.height;
        request.config.width = canvas.width;
    }

    if ((request.config.height > 0) != (request.config.width > 0))
        throw std::invalid_argument("MiniMax-H3 height and width must be supplied together");

    result.height =
        request.config.height > 0 ? request.config.height : kMiniMaxH3DefaultOutputHeight;
    result.width = request.config.width > 0 ? request.config.width : kMiniMaxH3DefaultOutputWidth;
    if (!is_minimax_h3_native_canvas(result.height, result.width)) {
        throw std::invalid_argument(
            "MiniMax-H3 output canvas must come from the public 768p resolver or be the explicit "
            "544x960/960x544 native profile");
    }

    // Make defaults explicit at the DSO boundary. This guarantees that later
    // output validation cannot drift from the profile the plugin generates.
    request.config.video_num_frames = result.num_frames;
    request.config.height = result.height;
    request.config.width = result.width;
    return result;
}

} // namespace trtmc::cli
