/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/reference_conditioner.h"
#include "families/minimax_h3/runtime/reference_visual_encoder.h"
#include "families/minimax_h3/runtime/video_decode_math.h"

#include <memory>
#include <mutex>

namespace trtmc::minimax_h3 {

enum class VideoDecoderResidency { Staged, Resident };

struct VideoDecoderConfig {
    int32_t latent_frames{37}, latent_height{84}, latent_width{48};
    int32_t tile_batch{1};
    ReferenceLatentNormalization normalization;
};

// Original VisualVAE tiling, temporal overlap and trailing crop, with variable
// portrait/landscape grids. The caller supplies generated-only [rows,96].
class VideoDecoder final {
  public:
    VideoDecoder(ReferenceModuleLoader loader, VideoDecoderConfig config,
                 cudaStream_t stream = nullptr,
                 VideoDecoderResidency residency = VideoDecoderResidency::Staged);
    DecodedH3Video run(const std::vector<float>& generated_rows);

  private:
    ReferenceModuleLoader loader_;
    VideoDecoderConfig config_;
    VideoDecodeGeometry geometry_;
    cudaStream_t stream_;
    VideoDecoderResidency residency_;
    std::unique_ptr<ITrtModule> module_;
    std::mutex mutex_;
};

} // namespace trtmc::minimax_h3
