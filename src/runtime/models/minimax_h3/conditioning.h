/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace trtmc {

struct MiniMaxH3PreparedKeyframes {
    std::vector<VideoImageInput> images;
    // Frame indices in the generated clip. The public FL2VA contract anchors
    // keyframes at frame zero and/or the last generated frame.
    std::vector<int32_t> anchors;
};

// Native, separable Lanczos-3 resize used by every H3 conditioning workflow.
// The result is always packed HWC RGB float32 in [0, 1].
VideoImageInput resize_minimax_h3_image_lanczos(const VideoImageInput& source,
                                                int32_t target_height, int32_t target_width);

// Implements the released FL2VA keyframe geometry. The first supplied
// keyframe is stretched to the target canvas. When both endpoints are
// supplied, the second is cover-resized and centre-cropped with the reference
// implementation's round/offset arithmetic.
MiniMaxH3PreparedKeyframes
prepare_minimax_h3_keyframes(const std::optional<VideoImageInput>& first_frame,
                             const std::optional<VideoImageInput>& last_frame,
                             int32_t target_height, int32_t target_width, int32_t output_frames);

// Ref2VA normalization helpers. Images use their own 2048-pixel short edge
// without an area cap. Videos keep their own aspect-ratio canvas, are mapped
// to the model's 24 fps clock by holding source frames, and are truncated to
// output_frames. Audio is truncated at its source rate, converted to stereo,
// and resampled once to 32 kHz.
VideoImageInput normalize_minimax_h3_reference_image(const VideoImageInput& source);
std::vector<int32_t> make_minimax_h3_reference_frame_map(int32_t source_frames,
                                                         int32_t fps_numerator,
                                                         int32_t fps_denominator,
                                                         int32_t output_frames);
VideoClipInput normalize_minimax_h3_reference_video(const VideoClipInput& source,
                                                    int32_t output_frames);
AudioResult normalize_minimax_h3_reference_audio(const AudioResult& source, int32_t output_frames);

// Validates the public reference-count/order contract and returns normalized
// entries in exactly the order supplied by the caller.
std::vector<VideoReferenceInput>
normalize_minimax_h3_references(const std::vector<VideoReferenceInput>& references,
                                int32_t output_frames);

} // namespace trtmc
