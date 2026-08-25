/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <cstdint>

namespace trtmc {

inline constexpr int32_t kMiniMaxH3ReferenceFps = 24;
inline constexpr int32_t kMiniMaxH3ReferenceAudioSampleRate = 32000;

struct MiniMaxH3ReferenceCanvas {
    int32_t height{0};
    int32_t width{0};
};

// Put one FL2VA keyframe on the target canvas. The geometry anchor is
// stretched; a follower is aspect-preserving cover-resized and center-cropped.
// Input and output pixels are HWC RGB float32 in [0, 1].
MediaImageInput minimax_h3_prepare_keyframe_image(const MediaImageInput& image,
                                                  int32_t target_height, int32_t target_width,
                                                  bool stretch);

// Normalize a reference video onto MiniMax-H3's 24 fps grid by dropping or
// duplicating whole THWC frames. Spatial pixels and any soundtrack are left
// unchanged.
MediaVideoInput minimax_h3_normalize_reference_video_fps(const MediaVideoInput& video);

// Resolve and prepare Ref2VA media at the exact reference-owned geometry.
// Images use a 2048-pixel short edge without an area cap. Videos use the
// 768-pixel short edge and 768x1344 soft area cap, after which both axes are
// rounded independently to a multiple of 32.
MiniMaxH3ReferenceCanvas minimax_h3_resolve_reference_image_canvas(int32_t source_height,
                                                                   int32_t source_width);
MiniMaxH3ReferenceCanvas minimax_h3_resolve_reference_video_canvas(int32_t source_height,
                                                                   int32_t source_width);
// Match Diffusers' `np.round(... * 255).astype(uint8) / 255` conversion for
// in-memory floating-point references, including half-to-even ties.
std::vector<float> minimax_h3_quantize_reference_pixels(const std::vector<float>& pixels);
MediaImageInput minimax_h3_prepare_reference_image(const MediaImageInput& image);
MediaVideoInput minimax_h3_prepare_reference_video(const MediaVideoInput& video,
                                                   int32_t max_frames);

// Snap a prepared video down to the largest 17*n+5 prefix the visual VAE can
// encode without temporal padding.
int32_t minimax_h3_trim_reference_num_frames(int32_t num_frames);

// Truncate a channel-major mono/stereo reference at its source rate, upmix mono
// to stereo, and resample once to 32 kHz with torchaudio's default
// sinc_interp_hann kernel. Native 32 kHz input remains an exact identity.
MultiChannelAudioResult minimax_h3_prepare_reference_audio(const MultiChannelAudioResult& audio,
                                                           double max_duration_seconds);

// Mirror AudioVAE.encode(): right-pad each 32 kHz stereo channel with zeros to
// the 800-sample hop before binding the dynamic reference-encoder plan.
MultiChannelAudioResult
minimax_h3_align_reference_audio_for_vae(const MultiChannelAudioResult& audio);

} // namespace trtmc
