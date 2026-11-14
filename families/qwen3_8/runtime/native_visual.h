/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/qwen3_8/runtime/pipeline.h"
#include "trtmc/internal/image.h"
#include "trtmc/internal/video.h"
#include "trtmc/runtime/trt_module.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::qwen3_8 {

struct NativeVisionConfig {
    int32_t patch{16}, temporal{2}, merge{2}, positions{2304}, head_dim{72}, hidden{0};
    int32_t min_tokens{128}, max_tokens{1280};
    float theta{10000};
    std::array<float, 3> mean{0.5F, 0.5F, 0.5F}, stddev{0.5F, 0.5F, 0.5F};
    void validate() const;
};

struct NativeVisualInputs {
    int32_t height{0}, width{0}, patch_width{0}, rotary_width{0}; // patch-grid dimensions
    std::vector<uint16_t> patches, position_weights;
    std::vector<float> rotary_angles;
    std::vector<int32_t> position_indices;
    TensorMap tensors();
};

// Pure host preprocessing. The temporal span contains exactly one group (the
// last source frame is repeated to fill an incomplete group).
std::array<int32_t, 2> native_visual_size(const NativeVisionConfig&, int32_t frames, bool video,
                                          int32_t height, int32_t width);
NativeVisualInputs native_visual_inputs(const NativeVisionConfig&,
                                        Span<const internal::ImageView> temporal_frames,
                                        int32_t height, int32_t width);

struct NativeVisualGroup {
    int32_t height, width; // merged token grid, one temporal patch group
    bool video;
    double timestamp;
    std::vector<uint16_t> features; // FP16 [height * width, text hidden]
};

// Expand one source media part. Videos retain the provider outer wrapper and
// per-temporal-group timestamps, wrappers and video-pad tokens.
std::string native_visual_text(const std::vector<NativeVisualGroup>& groups);

// Bind already expanded/tokenized image and video-frame spans to visual rows.
// All text tokens retain provider order; KV positions remain independent.
Qwen38MediaPrompt native_media_prompt(const std::vector<int32_t>& ids,
                                      const std::vector<NativeVisualGroup>& groups, int32_t hidden,
                                      DType dtype, int32_t image_token, int32_t video_token,
                                      int32_t vision_start, int32_t vision_end);

class NativeVisualEncoder {
  public:
    NativeVisualEncoder(std::unique_ptr<ITrtModule>, NativeVisionConfig);
    std::vector<NativeVisualGroup> encode(const internal::ImageView&);
    std::vector<NativeVisualGroup> encode(const internal::VideoView&);

  private:
    std::unique_ptr<ITrtModule> module_;
    NativeVisionConfig config_;
    std::vector<NativeVisualGroup> encode_frames(Span<const internal::ImageView>,
                                                 Span<const double>, bool video);
};

} // namespace trtmc::qwen3_8
