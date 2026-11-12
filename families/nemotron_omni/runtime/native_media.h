/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <vector>

namespace trtmc::nemotron_omni {
// Executes all modality encoders using MC's public TensorRT backend. The caller
// owns request tiling, frame selection and prompt ordering; no frames are dropped.
class NativeMedia {
  public:
    explicit NativeMedia(const FamilyContext& context);
    std::vector<float> visual(const std::vector<std::vector<float>>& frames, int height, int width);
    std::vector<float> audio(const std::vector<float>& pcm16k);
    int hidden() const noexcept { return hidden_; }
    int patch() const noexcept { return patch_; }
    int max_patches() const noexcept { return max_patches_; }
    int max_audio_frames() const noexcept { return max_audio_frames_; }

  private:
    std::unique_ptr<ITrtModule> image_patch_, video_patch_, visual_, audio_;
    std::vector<float> positions_, mel_filters_;
    int hidden_{0}, patch_{0}, position_side_{0}, visual_hidden_{0}, max_patches_{0};
    int audio_hidden_{0}, audio_heads_{0}, max_audio_frames_{0};
};
} // namespace trtmc::nemotron_omni
