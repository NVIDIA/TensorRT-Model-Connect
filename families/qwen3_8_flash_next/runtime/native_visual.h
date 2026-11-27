/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/image.h"
#include "trtmc/internal/video.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::qwen3_8_flash_next {

struct NativeVisionConfig {
    int32_t patch{16}, temporal{2}, merge{2}, positions{2304}, head_dim{72}, hidden{0};
    int32_t max_tokens{0}; // merged rows per temporal group covered by the plan
    int32_t image_min_pixels{65536}, image_max_pixels{16777216};
    int32_t video_min_pixels{4096}, video_max_pixels{25165824};
    float theta{10000};
    std::array<float, 3> mean{0.5F, 0.5F, 0.5F}, stddev{0.5F, 0.5F, 0.5F};
    void validate() const;
};

struct NativeVisualInputs {
    int32_t height{0}, width{0}, patch_width{0}, rotary_width{0}; // patch-grid dimensions
    std::vector<uint16_t> patches;
    std::vector<float> position_weights;
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

class NativeVisualEncoder {
  public:
    NativeVisualEncoder(std::unique_ptr<ITrtModule>, NativeVisionConfig);
    std::vector<NativeVisualGroup> encode(const internal::ImageView&, int32_t remaining_tokens);
    std::vector<NativeVisualGroup> encode(const internal::VideoView&, int32_t remaining_tokens);

  private:
    std::unique_ptr<ITrtModule> module_;
    NativeVisionConfig config_;
    std::vector<NativeVisualGroup> encode_frames(Span<const internal::ImageView>,
                                                 Span<const double>, bool video,
                                                 int32_t remaining_tokens);
};

} // namespace trtmc::qwen3_8_flash_next
