/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/visual_reference.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::minimax_h3 {

struct VideoDecodeGeometry {
    int32_t latent_frames, latent_height, latent_width;
    int32_t frames, height, width, clips, pad_tokens;
    VisualReferenceAxis vertical, horizontal;
    std::size_t tiles() const { return vertical.starts.size() * horizontal.starts.size(); }
};

VideoDecodeGeometry make_video_decode_geometry(int32_t latent_frames, int32_t latent_height,
                                               int32_t latent_width);

// IEEE binary16 round-to-nearest-even and lossless widening. These describe
// upstream host-assembly arithmetic, not a model precision option or GPU kernel.
uint16_t video_float_to_half(float value);
float video_half_to_float(uint16_t value);

struct DecodedH3Video {
    int32_t frames, height, width;
    std::vector<float> rgb; // Owned THWC FP32 [0,1], matching the shared contract.
    uint64_t forwards{0};   // Actual successful engine calls; zero for host-only assembly.
};

// Consumes original (unblended) tiles in clip/row-major order. Keeps only one
// raw tile row plus the five-frame temporal overlap, not every decoded clip or
// a second complete CTHW video. All upstream FP16 blend boundaries are explicit.
class VideoClipAssembler final {
  public:
    explicit VideoClipAssembler(VideoDecodeGeometry geometry);
    ~VideoClipAssembler();

    void append(int32_t clip, std::size_t tile, const float* decoded, std::size_t count);
    DecodedH3Video finish();

  private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace trtmc::minimax_h3
