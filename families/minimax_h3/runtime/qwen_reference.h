/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::minimax_h3 {

struct QwenImageGrid {
    int32_t height{0}; // 16-pixel patches, before 2x2 spatial merging.
    int32_t width{0};
};

struct QwenVisionMetadata {
    QwenImageGrid grid;
    std::vector<int32_t> position_indices; // [patches,4], learned 48x48 table.
    std::vector<float> position_weights;   // [patches,4], FP32 interpolation.
    std::vector<float> rotary_cos;         // [patches,72], FP32 rotate-half.
    std::vector<float> rotary_sin;
};

QwenVisionMetadata make_qwen_vision_metadata(QwenImageGrid grid);

// Input MUST already have the exact released processor's resize applied.
// This function does not resize, crop, change the reference resolution, or
// accept a VAE-normalized image. RGB uint8 is fused-normalized to [-1,1]
// and packed as [patches,3*2*16*16] in spatial-merge-block order.
std::vector<float> pack_resized_qwen_image(const uint8_t* rgb, std::size_t bytes, int32_t height,
                                           int32_t width);

struct QwenImagePresentation {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> h3_token_tags; // H3 video=0/text=1, NOT Qwen mm types.
    std::vector<int32_t> visual_mask;   // Native TensorMap int32 [rows,1].
    std::vector<int32_t> visual_rows;   // Pad positions, in image-feature order.
    std::vector<int32_t> position_ids;  // Qwen language [3,rows], axis-major.
    int32_t rope_delta{0};
};

// Image portion of the original Ref2VA presentation. No chat template/BOS,
// no unconditional branch. The tokenizer must have special insertion disabled.
// max_rows is the loaded conditioner profile: reject rather than truncate or
// downsample a reference to fit it. Audio/video presentations remain separate.
QwenImagePresentation make_qwen_image_presentation(const ITokenizer& tokenizer,
                                                   const std::string& prompt,
                                                   const std::vector<QwenImageGrid>& images,
                                                   int32_t max_rows);

} // namespace trtmc::minimax_h3
